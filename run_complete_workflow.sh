#!/bin/bash
# Script to run the complete workflow: setup, download data, pretrain, and train

set -e  # Exit on error

PROJECT_DIR="$(pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
DATA_DIR="${PROJECT_DIR}/data"
WEIGHTS_DIR="${PROJECT_DIR}/pretrained_weights"
TEACHER_CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/teacher_pretrain"
STUDENT_CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/student"
TEACHER_LOG_DIR="${PROJECT_DIR}/logs/teacher_pretrain"
STUDENT_LOG_DIR="${PROJECT_DIR}/logs/student"

# Setup environment
echo "===== Setting up environment ====="
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p "$DATA_DIR" "$WEIGHTS_DIR" "$TEACHER_CHECKPOINT_DIR" "$STUDENT_CHECKPOINT_DIR" "$TEACHER_LOG_DIR" "$STUDENT_LOG_DIR"

# Download dataset
echo "===== Downloading dataset ====="
python scripts/download_nyudepthv2_sample.py

# Download DINOv2 weights
echo "===== Downloading DINOv2 weights ====="
python scripts/download_dynov2_weights.py --model vit_base --output_dir "$WEIGHTS_DIR"

# Convert weights
echo "===== Converting weights ====="
python pretrain/convert_dynov2_weights.py \
    --input "${WEIGHTS_DIR}/dinov2_vit_base_pretrain.pth" \
    --output "${WEIGHTS_DIR}/dynov2_teacher_converted.pth"

# Pretrain teacher
echo "===== Pretraining teacher model ====="
python pretrain/pretrain_teacher.py \
    --config configs/pretrain_teacher.yaml \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$TEACHER_CHECKPOINT_DIR" \
    --log_dir "$TEACHER_LOG_DIR" \
    --pretrained_weights "${WEIGHTS_DIR}/dynov2_teacher_converted.pth" \
    --epochs 20  # Reduced for demonstration purposes

# Train teacher-student
echo "===== Training teacher-student model ====="
python train_teacher_student.py \
    --config configs/training.yaml \
    --teacher_weights "${TEACHER_CHECKPOINT_DIR}/model_best.pth.tar" \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$STUDENT_CHECKPOINT_DIR" \
    --log_dir "$STUDENT_LOG_DIR"

# Evaluate model
echo "===== Evaluating trained model ====="
python eval.py \
    --config configs/training.yaml \
    --model_path "${STUDENT_CHECKPOINT_DIR}/model_best.pth.tar" \
    --data_dir "$DATA_DIR" \
    --output_dir "./evaluation_results"

echo "===== Complete workflow finished! ====="
echo "Check evaluation results in './evaluation_results'"
