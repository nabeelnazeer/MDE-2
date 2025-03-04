"""
Script to prepare the Virtual KITTI 2.0 dataset for monocular depth estimation.
This script organizes RGB-depth pairs from Virtual KITTI 2.0 into train/val/test splits.
"""
import os
import shutil
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Virtual KITTI 2.0 dataset')
    parser.add_argument('--rgb_root', type=str, 
                        default='/Users/nabeelnazeer/Documents/Project-s6/Datasets/vkitti_2.0.3_rgb',
                        help='Root directory for RGB images')
    parser.add_argument('--depth_root', type=str,
                        default='/Users/nabeelnazeer/Documents/Project-s6/Datasets/vkitti_2.0.3_depth',
                        help='Root directory for depth maps')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/nabeelnazeer/Documents/Project-s6/MDE-2/data',
                        help='Output directory to save the prepared dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--max_samples_per_category', type=int, default=100,
                        help='Maximum number of samples per category (0 for all)')
    parser.add_argument('--resize', type=int, nargs=2, default=[256, 256],
                        help='Resize images to this resolution (width, height)')
    return parser.parse_args()

def find_scenes_and_categories(root_dir):
    """Find all scenes and their categories in the dataset"""
    scenes = []
    for scene_dir in sorted(os.listdir(root_dir)):
        if scene_dir.startswith('Scene'):
            scene_path = os.path.join(root_dir, scene_dir)
            if os.path.isdir(scene_path):
                categories = []
                for category_dir in sorted(os.listdir(scene_path)):
                    category_path = os.path.join(scene_path, category_dir)
                    if os.path.isdir(category_path):
                        categories.append(category_dir)
                scenes.append((scene_dir, categories))
    return scenes

def find_matching_pairs(rgb_root, depth_root, scene, category):
    """Find matching RGB-depth pairs for a given scene and category"""
    rgb_path = os.path.join(rgb_root, scene, category, "frames", "rgb", "Camera_0")
    depth_path = os.path.join(depth_root, scene, category, "frames", "depth", "Camera_0")
    
    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
        print(f"Warning: Missing path - RGB: {os.path.exists(rgb_path)}, Depth: {os.path.exists(depth_path)}")
        return []
    
    try:
        # Get all files
        rgb_files = sorted([f for f in os.listdir(rgb_path) if f.startswith('rgb_') and f.endswith('.jpg')])
        depth_files = sorted([f for f in os.listdir(depth_path) if f.startswith('depth_') and f.endswith('.png')])
        
        # Match pairs based on frame number
        pairs = []
        for rgb_file in rgb_files:
            # Extract frame number (e.g., '00000' from 'rgb_00000.jpg')
            frame_num = rgb_file.replace('rgb_', '').replace('.jpg', '')
            depth_file = f"depth_{frame_num}.png"
            
            if depth_file in depth_files:
                rgb_full = os.path.join(rgb_path, rgb_file)
                depth_full = os.path.join(depth_path, depth_file)
                pairs.append((rgb_full, depth_full))
        
        print(f"Found {len(pairs)} pairs for {scene}/{category}")
        return pairs
        
    except Exception as e:
        print(f"Error processing {scene}/{category}: {str(e)}")
        return []

def get_scene_categories(scene_path):
    """Get all valid categories for a scene"""
    categories = []
    for category in sorted(os.listdir(scene_path)):
        rgb_path = os.path.join(scene_path, category, "frames/rgb/Camera_0")
        if os.path.exists(rgb_path) and os.path.isdir(rgb_path):
            categories.append(category)
    return categories

def prepare_dataset(args):
    """Prepare the VKITTI dataset by creating train/val/test splits"""
    random.seed(args.seed)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'depths'), exist_ok=True)
    
    # Find all scenes
    scenes = sorted([d for d in os.listdir(args.rgb_root) 
                    if d.startswith('Scene') and os.path.isdir(os.path.join(args.rgb_root, d))])
    print(f"Found {len(scenes)} scenes: {', '.join(scenes)}")
    
    # Collect all pairs across all scenes and categories
    all_pairs = []
    
    # Process each scene
    for scene in scenes:
        print(f"\nProcessing scene: {scene}")
        scene_path = os.path.join(args.rgb_root, scene)
        
        # Get categories for this scene
        categories = sorted([d for d in os.listdir(scene_path) 
                           if os.path.isdir(os.path.join(scene_path, d))])
        print(f"Found categories: {', '.join(categories)}")
        
        # Process each category
        for category in categories:
            print(f"\nProcessing category: {category}")
            pairs = find_matching_pairs(args.rgb_root, args.depth_root, scene, category)
            
            if pairs:
                # Add scene and category info to pairs
                pairs = [(rgb, depth, scene, category) for rgb, depth in pairs]
                all_pairs.extend(pairs)
    
    # Print total pairs found
    print(f"\nTotal pairs found: {len(all_pairs)}")
    
    # Shuffle and split pairs
    random.shuffle(all_pairs)
    total_samples = len(all_pairs)
    train_end = int(total_samples * args.train_ratio)
    val_end = train_end + int(total_samples * args.val_ratio)
    
    splits = {
        'train': all_pairs[:train_end],
        'val': all_pairs[train_end:val_end],
        'test': all_pairs[val_end:]
    }
    
    # Process each split
    counters = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, split_pairs in splits.items():
        print(f"\nProcessing {split_name} split...")
        img_dir = os.path.join(args.output_dir, split_name, 'images')
        depth_dir = os.path.join(args.output_dir, split_name, 'depths')
        
        for idx, (rgb_path, depth_path, scene, category) in enumerate(tqdm(split_pairs)):
            try:
                # Load and process images
                rgb_img = Image.open(rgb_path).convert('RGB')
                depth_img = Image.open(depth_path)
                
                # Resize if specified
                if args.resize:
                    rgb_img = rgb_img.resize(tuple(args.resize), Image.BICUBIC)
                    depth_img = depth_img.resize(tuple(args.resize), Image.BICUBIC)
                
                # Save with new names
                rgb_save_path = os.path.join(img_dir, f"{idx:05d}.jpg")
                depth_save_path = os.path.join(depth_dir, f"{idx:05d}.png")
                
                rgb_img.save(rgb_save_path, quality=95)
                depth_img.save(depth_save_path)
                
                counters[split_name] += 1
                
            except Exception as e:
                print(f"Error processing pair {idx} in {split_name} split: {str(e)}")
                print(f"RGB path: {rgb_path}")
                print(f"Depth path: {depth_path}")
                continue
    
    # Print final statistics
    print("\nDataset preparation completed!")
    for split_name, count in counters.items():
        print(f"{split_name}: {count} samples")

def process_and_copy_pair(rgb_path, depth_path, img_out_dir, depth_out_dir, filename, resize=None):
    """Process and copy an RGB-depth pair to the output directories"""
    # Load images
    rgb_img = Image.open(rgb_path).convert('RGB')
    depth_img = Image.open(depth_path)
    
    # Resize if specified
    if resize:
        rgb_img = rgb_img.resize(resize, Image.BICUBIC)
        depth_img = depth_img.resize(resize, Image.BICUBIC)
    
    # Save images
    rgb_out_path = os.path.join(img_out_dir, f"{filename}.jpg")
    depth_out_path = os.path.join(depth_out_dir, f"{filename}.png")
    
    rgb_img.save(rgb_out_path, quality=95)
    depth_img.save(depth_out_path)

def main():
    args = parse_args()
    prepare_dataset(args)

if __name__ == '__main__':
    main()
