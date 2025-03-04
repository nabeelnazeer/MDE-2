"""
Script to analyze the Virtual KITTI 2.0 dataset structure and print information.
This helps understand the dataset organization before preparing it for training.
"""
import os
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Virtual KITTI 2.0 dataset')
    parser.add_argument('--rgb_root', type=str, 
                        default='/Users/nabeelnazeer/Documents/Project-s6/Datasets/vkitti_2.0.3_rgb',
                        help='Root directory for RGB images')
    parser.add_argument('--depth_root', type=str,
                        default='/Users/nabeelnazeer/Documents/Project-s6/Datasets/vkitti_2.0.3_depth',
                        help='Root directory for depth maps')
    return parser.parse_args()

def analyze_dataset(rgb_root, depth_root):
    """Analyze the Virtual KITTI 2.0 dataset structure"""
    # Find all scenes
    scenes = [d for d in sorted(os.listdir(depth_root)) if d.startswith('Scene') and os.path.isdir(os.path.join(depth_root, d))]
    print(f"Found {len(scenes)} scenes: {', '.join(scenes)}")
    
    # Statistics
    total_rgb_images = 0
    total_depth_images = 0
    categories_per_scene = defaultdict(list)
    image_counts = defaultdict(lambda: defaultdict(int))
    image_sizes = []
    depth_ranges = []
    
    # Analyze each scene
    for scene in scenes:
        print(f"\nAnalyzing scene: {scene}")
        scene_path = os.path.join(depth_root, scene)
        
        # Find all categories in the scene
        categories = [d for d in sorted(os.listdir(scene_path)) if os.path.isdir(os.path.join(scene_path, d))]
        categories_per_scene[scene] = categories
        print(f"  Found {len(categories)} categories: {', '.join(categories)}")
        
        # Analyze each category
        for category in categories:
            rgb_dir = os.path.join(rgb_root, scene, category, "frames", "rgb", "Camera_0")
            depth_dir = os.path.join(depth_root, scene, category, "frames", "depth", "Camera_0")
            
            # Check if directories exist
            if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
                print(f"  Warning: Missing directory for {scene}/{category}")
                continue
            
            # Count images
            rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')]
            depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png') or f.endswith('.jpg')]
            
            image_counts[scene][category] = len(rgb_files)
            total_rgb_images += len(rgb_files)
            total_depth_images += len(depth_files)
            
            # Analyze image and depth properties (for a sample)
            if rgb_files and depth_files:
                # Sample the first image for stats
                rgb_sample = os.path.join(rgb_dir, rgb_files[0])
                depth_sample = os.path.join(depth_dir, depth_files[0])
                
                # Get image size
                with Image.open(rgb_sample) as img:
                    image_sizes.append(img.size)
                
                # Get depth range
                with Image.open(depth_sample) as img:
                    depth_array = np.array(img)
                    depth_ranges.append((np.min(depth_array), np.max(depth_array)))
    
    # Print overall statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total RGB images: {total_rgb_images}")
    print(f"Total depth maps: {total_depth_images}")
    
    # Image sizes
    if image_sizes:
        unique_sizes = set(image_sizes)
        print(f"Image sizes: {', '.join([f'{w}x{h}' for w, h in unique_sizes])}")
    
    # Depth ranges
    if depth_ranges:
        min_depth = min([r[0] for r in depth_ranges])
        max_depth = max([r[1] for r in depth_ranges])
        print(f"Depth range: {min_depth} to {max_depth}")
    
    # Print detailed image counts
    print("\n=== Image Counts per Scene/Category ===")
    for scene, categories in categories_per_scene.items():
        print(f"{scene}:")
        for category in categories:
            count = image_counts[scene].get(category, 0)
            print(f"  {category}: {count} images")
    
    # Additional analysis if needed
    print("\n=== Category Analysis ===")
    all_categories = set()
    for categories in categories_per_scene.values():
        all_categories.update(categories)
    
    print(f"All categories across scenes: {', '.join(sorted(all_categories))}")
    
    # Check for consistency across scenes
    for category in sorted(all_categories):
        scenes_with_category = [scene for scene, cats in categories_per_scene.items() if category in cats]
        if len(scenes_with_category) < len(scenes):
            print(f"Category '{category}' is missing in scenes: {', '.join(set(scenes) - set(scenes_with_category))}")

def main():
    args = parse_args()
    analyze_dataset(args.rgb_root, args.depth_root)

if __name__ == '__main__':
    main()
