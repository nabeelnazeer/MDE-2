"""
Script to pretrain the DynoV2 teacher model for monocular depth estimation.
This pretraining focuses specifically on adapting the model for depth tasks.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.teacher import DynoV2Teacher
from data.dataloader import DepthDataset
from losses.scale_shift_invariant import MultiScaleScaleShiftInvariantLoss
from utils import save_checkpoint, visualize_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain DynoV2 teacher model for depth estimation')
    parser.add_argument('--config', type=str, default='configs/pretrain_teacher.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/teacher_pretrain', 
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/teacher_pretrain', 
                        help='Directory to save logs')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Path to pretrained DynoV2 weights')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--freeze_encoder', action='store_true', 
                        help='Freeze encoder backbone')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to resume training from checkpoint')
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch") as tepoch:
        for images, targets in tepoch:
            # Move to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            tepoch.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / num_batches

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation", unit="batch") as tepoch:
            for images, targets in tepoch:
                # Move to device
                images = images.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                tepoch.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / num_batches

def main():
    args = parse_args()
    
    # Load config if exists
    config = vars(args)
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Update config with yaml values
            for k, v in yaml_config.items():
                if k not in config or config[k] is None:
                    config[k] = v
    
    # Create output directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=config['log_dir'])
    
    # Create model
    model = DynoV2Teacher(pretrained=False, freeze_encoder=config['freeze_encoder']).to(device)
    
    # Load pretrained weights if provided
    if config['pretrained_weights']:
        print(f"Loading pretrained weights from {config['pretrained_weights']}")
        checkpoint = torch.load(config['pretrained_weights'], map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
            
        # Load weights with flexible strictness
        model.load_state_dict(checkpoint, strict=False)
        print("Pretrained weights loaded")
    
    # Define loss function
    criterion = MultiScaleScaleShiftInvariantLoss(
        alpha=config.get('ssi_alpha', 0.5),
        scales=config.get('ssi_scales', 4)
    ).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate']
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=5, verbose=True
    )
    
    # Create dataloaders
    train_dataset = DepthDataset(
        data_dir=config['data_dir'],
        split='train',
        transform=None,  # Use default transform
        target_transform=None  # Use default transform
    )
    
    val_dataset = DepthDataset(
        data_dir=config['data_dir'],
        split='val',
        transform=None,  # Use default transform
        target_transform=None  # Use default transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Resume training if checkpoint provided
    start_epoch = 0
    best_val_loss = float('inf')
    if config['resume']:
        if os.path.isfile(config['resume']):
            print(f"Loading checkpoint '{config['resume']}'")
            checkpoint = torch.load(config['resume'], map_location=device)
            
            # Load checkpoint data
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            print(f"Loaded checkpoint '{config['resume']}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{config['resume']}'")
    
    # Training loop
    print(f"Starting training for {config['epochs']} epochs")
    
    for epoch in range(start_epoch, config['epochs']):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config['epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, config['checkpoint_dir'], f"checkpoint_{epoch+1}.pth.tar")
        
        # Visualize predictions
        if (epoch + 1) % config.get('visualize_every', 5) == 0:
            vis_dir = os.path.join(config['log_dir'], f'vis_epoch_{epoch+1}')
            visualize_predictions(
                model, val_loader, device, vis_dir, 
                num_samples=config.get('num_vis_samples', 5)
            )
    
    # Save final model
    torch.save(model.state_dict(), 
               os.path.join(config['checkpoint_dir'], 'teacher_final.pth'))
    
    print("Training completed!")
    writer.close()

if __name__ == '__main__':
    main()
