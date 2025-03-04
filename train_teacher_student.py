"""
Training script for the teacher-student monocular depth estimation model.
This script trains the student model using supervision from both the ground truth and teacher model.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from tqdm import tqdm

from models.teacher import DynoV2Teacher
from models.student import StudentModel
from losses.scale_shift_invariant import MultiScaleScaleShiftInvariantLoss
from losses.gradient_matching import MultiScaleGradientMatchingLoss
from data.dataloader import get_dataloaders
from utils import save_checkpoint, evaluate_model, visualize_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Train teacher-student depth estimation model')
    parser.add_argument('--config', type=str, default='configs/training.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--teacher_weights', type=str, required=True,
                        help='Path to pretrained teacher weights')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/student',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/student',
                        help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}. Using default configuration.")
        from config import config
    
    # Override config with command line arguments
    config['teacher_weights'] = args.teacher_weights
    config['data_dir'] = args.data_dir
    config['checkpoint_dir'] = args.checkpoint_dir
    config['log_dir'] = args.log_dir
    
    # Create output directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=config['log_dir'])
    
    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Initialize models
    print("Initializing teacher model...")
    teacher_model = DynoV2Teacher(pretrained=False, freeze_encoder=True).to(device)
    
    # Load teacher weights
    print(f"Loading teacher weights from {config['teacher_weights']}")
    teacher_checkpoint = torch.load(config['teacher_weights'], map_location=device)
    if 'state_dict' in teacher_checkpoint:
        teacher_model.load_state_dict(teacher_checkpoint['state_dict'])
    else:
        teacher_model.load_state_dict(teacher_checkpoint)
    teacher_model.eval()  # Teacher model is always in eval mode
    
    print("Initializing student model...")
    student_model = StudentModel(in_channels=3).to(device)
    
    # Initialize loss functions
    print("Initializing loss functions...")
    ssi_loss = MultiScaleScaleShiftInvariantLoss(
        alpha=config['ssi_alpha'],
        scales=config['ssi_scales']
    ).to(device)
    
    gradient_loss = MultiScaleGradientMatchingLoss(
        scales=config['gradient_scales']
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume training if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            student_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    print(f"Starting training for {config['num_epochs']} epochs")
    total_steps = 0
    
    for epoch in range(start_epoch, config['num_epochs']):
        student_model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}") as tepoch:
            for images, depths in tepoch:
                images = images.to(device)
                depths = depths.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    teacher_preds = teacher_model(images)
                
                student_preds = student_model(images)
                student_features = student_model.get_multi_scale_features(images)
                
                # Calculate losses
                # 1. Scale-shift invariant loss between student and ground truth
                ssi_loss_val = ssi_loss(student_preds, depths)
                
                # 2. Scale-shift invariant loss between student and teacher
                teacher_student_loss = ssi_loss(student_preds, teacher_preds)
                
                # 3. Multi-scale gradient matching loss
                grad_loss_val = gradient_loss(student_features, depths)
                
                # Combine losses
                loss = (
                    config['lambda_ssi'] * ssi_loss_val +
                    config['lambda_teacher'] * teacher_student_loss +
                    config['lambda_gradient'] * grad_loss_val
                )
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=f"{loss.item():.4f}")
                
                # Log to tensorboard
                writer.add_scalar('Loss/train', loss.item(), total_steps)
                writer.add_scalar('Loss/ssi', ssi_loss_val.item(), total_steps)
                writer.add_scalar('Loss/teacher_student', teacher_student_loss.item(), total_steps)
                writer.add_scalar('Loss/gradient', grad_loss_val.item(), total_steps)
                
                total_steps += 1
        
        # Validation
        val_loss = evaluate_model(
            student_model, val_loader, ssi_loss, device
        )
        
        # Log validation loss
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss
            }, is_best=True, checkpoint_dir=config['checkpoint_dir'])
            
            # Visualize some predictions
            if epoch % config['visualize_every'] == 0:
                visualize_predictions(
                    student_model, val_loader, device, 
                    os.path.join(config['log_dir'], f'viz_epoch_{epoch+1}'),
                    num_samples=5
                )
        
        # Regular checkpoint saving
        if (epoch + 1) % config['save_every'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss
            }, is_best=False, checkpoint_dir=config['checkpoint_dir'], 
               filename=f'checkpoint_epoch_{epoch+1}.pth.tar')
    
    # Close tensorboard writer
    writer.close()
    
    print("Training completed!")

if __name__ == '__main__':
    main()
