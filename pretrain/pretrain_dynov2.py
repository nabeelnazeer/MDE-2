"""
Pretraining script for DynoV2 on depth data.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

from models.teacher import DynoV2Teacher
from data.dataloader import DepthDataset, get_dataloaders
from utils import compute_metrics, visualize_predictions
from losses.scale_shift_invariant import ScaleShiftInvariantLoss
from losses.gradient_matching import GradientLoss
from losses.combined import CombinedLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain DynoV2 on depth data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory to save logs')
    return parser.parse_args()

def evaluate_teacher(model, dataloader, device):
    """Detailed evaluation of teacher model"""
    model.eval()
    metrics = {
        'abs_rel': 0.0,
        'rmse': 0.0,
        'log_rmse': 0.0,
        'sq_rel': 0.0,
        'delta1': 0.0,
        'delta2': 0.0,
        'delta3': 0.0
    }
    total_samples = 0
    
    with torch.no_grad():
        for images, depths in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            depths = depths.to(device)
            
            predictions = model(images)
            
            # Compute metrics for each sample in batch
            for i in range(images.size(0)):
                pred = predictions[i:i+1]
                target = depths[i:i+1]
                
                batch_metrics = compute_metrics(pred, target)
                for k, v in batch_metrics.items():
                    metrics[k] += v
                
            total_samples += images.size(0)
    
    # Average metrics
    for k in metrics.keys():
        metrics[k] /= total_samples
    
    return metrics

def print_metrics(metrics, prefix=""):
    """Print depth estimation metrics in a formatted way"""
    print(f"\n{prefix} Depth Estimation Metrics:")
    print("-" * 50)
    print(f"Abs Relative Error: {metrics['abs_rel']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"RMSE log: {metrics['log_rmse']:.4f}")
    print(f"Sq Relative Error: {metrics['sq_rel']:.4f}")
    print(f"Accuracy (δ < 1.25): {metrics['delta1']:.4f}")
    print(f"Accuracy (δ < 1.25²): {metrics['delta2']:.4f}")
    print(f"Accuracy (δ < 1.25³): {metrics['delta3']:.4f}")

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device based on availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple M-series GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create model and move to device
    model = DynoV2Teacher(
        pretrained=True,
        freeze_encoder=False,
        arch='vit_base'
    ).to(device)
    
    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(args.data_dir, batch_size=8)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Initialize combined loss
    criterion = CombinedLoss(
        ssi_weight=1.0,
        gradient_weight=0.5,
        ssi_alpha=0.5,
        ssi_scales=4
    ).to(device)
    
    best_val_loss = float('inf')
    writer = SummaryWriter(log_dir=args.log_dir)
    
    print("\nStarting training...")
    print(f"Training on device: {device}")
    print(f"Total epochs: 50")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(50):
        model.train()
        train_metrics = {'total': 0.0, 'ssi': 0.0, 'gradient': 0.0}
        
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/50] Training')
        for batch_idx, (images, depths) in enumerate(pbar):
            try:
                images = images.to(device)
                depths = depths.to(device)
                
                # Clear existing gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                outputs = model(images)
                
                # Compute losses - handle any potential NaN
                losses = criterion(outputs, depths)
                if torch.isnan(losses['loss']):
                    print(f"NaN loss detected at batch {batch_idx}. Skipping...")
                    continue
                
                # Backward pass with gradient clipping
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update metrics
                for k, v in losses.items():
                    if k in train_metrics:
                        train_metrics[k] += v.item()
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'Loss': f"{losses['loss'].item():.4f}",
                        'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
                    })
                    
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Average metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
            writer.add_scalar(f'Loss/train_{k}', train_metrics[k], epoch)
        
        # Validation
        model.eval()
        val_metrics = {'total': 0.0, 'ssi': 0.0, 'gradient': 0.0}
        val_pbar = tqdm(val_loader, desc='Validation')
        
        with torch.no_grad():
            for images, depths in val_pbar:
                images = images.to(device)
                depths = depths.to(device)
                
                outputs = model(images)
                val_losses = criterion(outputs, depths)
                
                val_metrics['total'] += val_losses['loss'].item()
                val_metrics['ssi'] += val_losses['ssi_loss'].item()
                val_metrics['gradient'] += val_losses['gradient_loss'].item()
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'Loss': f"{val_losses['loss'].item():.4f}"
                })
        
        # Average validation metrics
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
            writer.add_scalar(f'Loss/val_{k}', val_metrics[k], epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['total'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print detailed epoch summary
        print(f"\nEpoch {epoch+1}/50 Summary:")
        print("-" * 50)
        print("Training Metrics:")
        print(f"  Total Loss: {train_metrics['total']:.4f}")
        print(f"  SSI Loss: {train_metrics['ssi']:.4f}")
        print(f"  Gradient Loss: {train_metrics['gradient']:.4f}")
        print("\nValidation Metrics:")
        print(f"  Total Loss: {val_metrics['total']:.4f}")
        print(f"  SSI Loss: {val_metrics['ssi']:.4f}")
        print(f"  Gradient Loss: {val_metrics['gradient']:.4f}")
        print(f"\nLearning Rate: {current_lr:.6f}")
        
        # Print detailed metrics every 5 epochs and on final epoch
        if (epoch + 1) % 5 == 0 or epoch == 49:
            print(f"\n{'='*20} Detailed Evaluation {'='*20}")
            print(f"Epoch [{epoch+1}/50]")
            
            # Evaluate on validation set
            val_depth_metrics = evaluate_teacher(model, val_loader, device)
            print_metrics(val_depth_metrics, prefix="Validation")
            
            # Log metrics to tensorboard
            for k, v in val_depth_metrics.items():
                writer.add_scalar(f'Metrics/{k}', v, epoch)
        
        # Save if best model and print metrics
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            save_path = os.path.join(args.checkpoint_dir, 'teacher_best.pth')
            
            # Get detailed metrics for best model
            best_metrics = evaluate_teacher(model, val_loader, device)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'depth_metrics': best_metrics
            }, save_path)
            
            print("\n" + "="*20 + " New Best Model " + "="*20)
            print(f"Epoch {epoch+1}")
            print_metrics(best_metrics, prefix="Best Model")
        
        # Regular checkpoint save
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.checkpoint_dir, f'teacher_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, save_path)

    # Final evaluation
    print("\n" + "="*20 + " Final Model Evaluation " + "="*20)
    final_metrics = evaluate_teacher(model, val_loader, device)
    print_metrics(final_metrics, prefix="Final")
    
    # Save final model with metrics
    final_save_path = os.path.join(args.checkpoint_dir, 'teacher_final.pth')
    torch.save({
        'epoch': 50,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_metrics': evaluate_teacher(model, val_loader, device)
    }, final_save_path)
    
    # Print training summary
    print("\n" + "="*20 + " Training Summary " + "="*20)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    writer.close()
    print("Pretraining completed!")

if __name__ == '__main__':
    main()
