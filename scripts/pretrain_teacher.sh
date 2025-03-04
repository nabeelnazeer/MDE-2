#!/bin/bash
# Script to download weights and pretrain the teacher model

# Set up directories
PROJECT_DIR="$(pwd)"
WEIGHTS_DIR="${PROJECT_DIR}/pretrained_weights"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/teacher_pretrain"
LOG_DIR="${PROJECT_DIR}/logs/teacher_pretrain"
CONFIG_FILE="${PROJECT_DIR}/configs/pretrain_teacher.yaml"
DATA_DIR="${PROJECT_DIR}/data"

# Create directories
mkdir -p "$WEIGHTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

# Step 1: Download pretrained DINOv2 weights
echo "Downloading pretrained DINOv2 weights..."
python scripts/download_dynov2_weights.py --model vit_base --output_dir "$WEIGHTS_DIR"

# Step 2: Convert weights to the format needed by our teacher model
echo "Converting weights for the teacher model..."
python pretrain/convert_dynov2_weights.py \
    --input "${WEIGHTS_DIR}/dinov2_vit_base_pretrain.pth" \
    --output "${WEIGHTS_DIR}/dynov2_teacher_converted.pth"

# Step 3: Pretrain the teacher model on depth data
echo "Pretraining the teacher model..."
python pretrain/pretrain_teacher.py \
    --config "$CONFIG_FILE" \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --pretrained_weights "${WEIGHTS_DIR}/dynov2_teacher_converted.pth" \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 0.0001

echo "Pretraining complete!"
