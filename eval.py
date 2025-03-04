import os
import torch
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.student import StudentModel
from data.dataloader import get_dataloaders
from utils import load_model, compute_metrics, visualize_predictions

def evaluate(model, dataloader, device, output_dir=None, save_visualizations=True, num_vis_samples=10):
    """Evaluate the model on the given dataloader"""
    model.eval()
    metrics_sum = {
        'rmse': 0.0,
        'mae': 0.0,
        'rel': 0.0,
        'delta1': 0.0,
        'delta2': 0.0,
        'delta3': 0.0
    }
    sample_count = 0
    
    # Create output directory for visualizations
    if output_dir is not None and save_visualizations:
        os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc='Evaluating')):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute metrics for batch
            batch_size = images.size(0)
            for i in range(batch_size):
                pred = predictions[i:i+1]
                target = targets[i:i+1]
                
                # Compute metrics
                batch_metrics = compute_metrics(pred, target)
                
                # Update metrics sum
                for metric, value in batch_metrics.items():
                    metrics_sum[metric] += value
            
            sample_count += batch_size
            
            # Save visualizations for the first few batches
            if save_visualizations and batch_idx < num_vis_samples // batch_size and output_dir is not None:
                for i in range(min(batch_size, num_vis_samples - batch_idx * batch_size)):
                    # Create individual visualization
                    img = images[i].cpu()
                    target = targets[i].cpu()
                    pred = predictions[i].cpu()
                    
                    # Save visualization
                    idx = batch_idx * batch_size + i
                    save_path = os.path.join(output_dir, f'sample_{idx}.png')
                    
                    # Create a single visualization
                    # We're creating a mini batch of 1 to use the visualization function
                    img_batch = img.unsqueeze(0)
                    target_batch = target.unsqueeze(0)
                    pred_batch = pred.unsqueeze(0)
                    
                    # Save to a temporary directory and then rename
                    tmp_dir = os.path.join(output_dir, 'tmp')
                    os.makedirs(tmp_dir, exist_ok=True)
                    
                    visualize_predictions(
                        lambda x: pred_batch, 
                        [(img_batch, target_batch)], 
                        device, 
                        tmp_dir, 
                        num_samples=1
                    )
                    
                    # Rename the file
                    tmp_file = os.path.join(tmp_dir, 'sample_1.png')
                    if os.path.exists(tmp_file):
                        os.rename(tmp_file, save_path)
    
    # Calculate average metrics
    for metric in metrics_sum:
        metrics_sum[metric] /= sample_count
    
    return metrics_sum

def main():
    parser = argparse.ArgumentParser(description='Evaluate depth estimation model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory (overrides config)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--no_vis', action='store_true', help='Disable visualization saving')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and load weights
    model = StudentModel(in_channels=3).to(device)
    model = load_model(model, args.model_path, device)
    
    # Get dataloaders
    _, val_loader, test_loader = get_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=args.num_workers
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_metrics = evaluate(
        model, 
        val_loader, 
        device, 
        output_dir=os.path.join(args.output_dir, 'validation'),
        save_visualizations=not args.no_vis
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate(
        model, 
        test_loader, 
        device, 
        output_dir=os.path.join(args.output_dir, 'test'),
        save_visualizations=not args.no_vis
    )
    
    # Print metrics
    print("\nValidation Results:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write("Validation Results:\n")
        for metric, value in val_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nTest Results:\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

if __name__ == '__main__':
    main()
