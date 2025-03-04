"""
Script to visualize RGB-depth pairs from the prepared Virtual KITTI 2.0 dataset.
This helps confirm that the dataset was prepared correctly.
"""
import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Virtual KITTI 2.0 samples')
    parser.add_argument('--data_dir', type=str,
                        default='/Users/nabeelnazeer/Documents/Project-s6/MDE-2/data',
                        help='Directory containing the prepared dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize per split')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/nabeelnazeer/Documents/Project-s6/MDE-2/data/visualization',
                        help='Directory to save visualizations')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='all',
                        help='Which dataset split to visualize')
    return parser.parse_args()

def visualize_samples(data_dir, output_dir, split='all', num_samples=5):
    """Visualize samples from the prepared dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ['train', 'val', 'test'] if split == 'all' else [split]
    
    for current_split in splits:
        print(f"Visualizing samples from {current_split} split")
        
        img_dir = os.path.join(data_dir, current_split, 'images')
        depth_dir = os.path.join(data_dir, current_split, 'depths')
        
        # Get all file pairs
        image_files = [f for f in sorted(os.listdir(img_dir)) if f.endswith('.jpg') or f.endswith('.png')]
        
        if not image_files:
            print(f"No images found in {img_dir}")
            continue
        
        # Select random samples
        if len(image_files) <= num_samples:
            selected_files = image_files
        else:
            selected_files = random.sample(image_files, num_samples)
        
        # Create figure for all samples in this split
        plt.figure(figsize=(15, 5 * num_samples))
        
        for i, img_file in enumerate(selected_files):
            # Get corresponding depth file
            depth_file = img_file.replace('.jpg', '.png')
            
            img_path = os.path.join(img_dir, img_file)
            depth_path = os.path.join(depth_dir, depth_file)
            
            # Skip if depth file doesn't exist
            if not os.path.exists(depth_path):
                print(f"Warning: Missing depth file for {img_file}")
                continue
            
            # Load images
            img = Image.open(img_path).convert('RGB')
            depth = Image.open(depth_path)
            
            # Convert to numpy arrays
            img_array = np.array(img)
            depth_array = np.array(depth)
            
            # Normalize depth for visualization
            depth_viz = depth_array.astype(np.float32)
            depth_viz = (depth_viz - depth_viz.min()) / (depth_viz.max() - depth_viz.min() + 1e-6)
            
            # Plot
            plt.subplot(num_samples, 2, i*2+1)
            plt.imshow(img_array)
            plt.title(f"RGB - {img_file}")
            plt.axis('off')
            
            plt.subplot(num_samples, 2, i*2+2)
            plt.imshow(depth_viz, cmap='viridis')
            plt.title(f"Depth - {depth_file}")
            plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{current_split}_samples.png"))
        plt.close()
        
        print(f"Visualizations saved to {os.path.join(output_dir, f'{current_split}_samples.png')}")

def main():
    args = parse_args()
    visualize_samples(
        args.data_dir, 
        args.output_dir,
        args.split,
        args.num_samples
    )

if __name__ == '__main__':
    main()
