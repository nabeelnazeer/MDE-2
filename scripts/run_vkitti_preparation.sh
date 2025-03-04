#!/bin/bash
# Script to prepare the Virtual KITTI 2.0 dataset for the MDE-2 project

# Set paths
RGB_ROOT="/Users/nabeelnazeer/Documents/Project-s6/Datasets/vkitti_2.0.3_rgb"
DEPTH_ROOT="/Users/nabeelnazeer/Documents/Project-s6/Datasets/vkitti_2.0.3_depth"
OUTPUT_DIR="/Users/nabeelnazeer/Documents/Project-s6/MDE-2/data"

# First, analyze the dataset structure
echo "Analyzing Virtual KITTI 2.0 dataset structure..."
python scripts/vkitti_dataset_info.py \
    --rgb_root "$RGB_ROOT" \
    --depth_root "$DEPTH_ROOT"

# Now prepare the dataset
echo -e "\n\nPreparing dataset for training..."
python scripts/prepare_vkitti_dataset.py \
    --rgb_root "$RGB_ROOT" \
    --depth_root "$DEPTH_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --max_samples_per_category 200 \
    --resize 256 256

echo "Dataset preparation complete!"
echo "Data is now ready at $OUTPUT_DIR for training the monocular depth estimation model."
