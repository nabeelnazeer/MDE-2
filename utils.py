import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """Save checkpoint to disk"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'model_best.pth.tar'))

def evaluate_model(model, dataloader, criterion, device, max_samples=None):
    """Evaluate model on dataloader"""
    model.eval()
    total_loss = 0.0
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_samples and batch_idx * images.size(0) >= max_samples:
                break
                
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Update metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
    
    # Calculate average loss
    avg_loss = total_loss / sample_count if sample_count > 0 else float('inf')
    
    return avg_loss

def visualize_predictions(model, dataloader, device, output_dir, num_samples=5):
    """
    Visualize depth predictions
    model: trained model
    dataloader: dataloader containing images and ground truth
    device: device to run model on
    output_dir: directory to save visualizations
    num_samples: number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Get predictions
            images = images.to(device)
            predictions = model(images)
            
            # Convert to numpy arrays
            image = images[0].cpu().numpy().transpose(1, 2, 0)
            target = targets[0].cpu().numpy().squeeze()
            pred = predictions[0].cpu().numpy().squeeze()
            
            # Normalize for visualization
            image = (image - image.min()) / (image.max() - image.min())
            target = (target - target.min()) / (target.max() - target.min())
            pred = (pred - pred.min()) / (pred.max() - pred.min())
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(target, cmap='magma')
            axes[1].set_title('Ground Truth Depth')
            axes[1].axis('off')
            
            axes[2].imshow(pred, cmap='magma')
            axes[2].set_title('Predicted Depth')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'))
            plt.close()

def compute_metrics(pred, target, mask=None):
    """
    Compute depth estimation metrics
    pred: predicted depth map
    target: ground truth depth map
    mask: optional mask for valid depth values
    """
    if mask is None:
        mask = target > 0
    
    pred = pred[mask]
    target = target[mask]
    
    # Avoid division by zero
    pred = pred + 1e-6
    target = target + 1e-6
    
    # Compute metrics
    thresh = torch.max((target / pred), (pred / target))
    delta1 = (thresh < 1.25).float().mean()
    delta2 = (thresh < 1.25 ** 2).float().mean()
    delta3 = (thresh < 1.25 ** 3).float().mean()
    
    rmse = torch.sqrt(((pred - target) ** 2).mean())
    log_rmse = torch.sqrt(((torch.log(pred) - torch.log(target)) ** 2).mean())
    abs_rel = torch.mean(torch.abs(pred - target) / target)
    sq_rel = torch.mean(((pred - target) ** 2) / target)
    
    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'log_rmse': log_rmse.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item()
    }

def load_model(model, checkpoint_path, device=None):
    """
    Load model weights from checkpoint
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Load state dict
        model.load_state_dict(state_dict)
        return model
    else:
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")