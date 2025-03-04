#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/train/images data/train/depths
mkdir -p data/val/images data/val/depths
mkdir -p data/test/images data/test/depths
mkdir -p checkpoints/teacher_pretrain checkpoints/student
mkdir -p logs/teacher_pretrain logs/student
mkdir -p pretrained_weights

echo "Setup completed. Activate the environment with: source .venv/bin/activate"
