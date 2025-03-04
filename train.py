import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import time
from tqdm import tqdm

from models.teacher import DynoV2Teacher
from models.student import StudentModel
from losses.scale_shift_invariant import MultiScaleScaleShiftInvariantLoss
from losses.gradient_matching import MultiScaleGradientMatchingLoss
from data.dataloader import get_dataloaders
from utils import save_checkpoint, evaluate_model, visualize_predictions

def train(config):
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
    teacher_model = DynoV2Teacher(pretrained=True, freeze_encoder=True).to(device)
    teacher_model.eval()  # Teacher model is always in eval mode
    
    student_model = StudentModel(in_channels=3).to(device)
    
    # Initialize loss functions
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
    
    # Training loop
    best_val_loss = float('inf')
    total_steps = 0
    
    for epoch in range(config['num_epochs']):
        student_model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{config['num_epochs']}")
            
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
                tepoch.set_postfix(loss=loss.item())
                
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train depth estimation model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Start training
    train(config)
