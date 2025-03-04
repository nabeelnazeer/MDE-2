#!/bin/bash

# Add project root to PYTHONPATH
export PYTHONPATH="/Users/nabeelnazeer/Documents/Project-s6/MDE-2:${PYTHONPATH}"

# Set paths
CHECKPOINT_DIR="./checkpoints"
VKITTI_DATA="./data"

# Pretrain directly on VKITTI
echo "=== Pretraining DynoV2 on VKITTI ==="
python3 pretrain/pretrain_dynov2.py \
    --config configs/pretrain_teacher.yaml \
    --data_dir $VKITTI_DATA \
    --checkpoint_dir "${CHECKPOINT_DIR}/teacher_pretrain" \
    --log_dir "./logs/teacher_pretrain"

echo "Pretraining completed!"
