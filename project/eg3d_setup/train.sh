#!/bin/bash

# EG3D Training Script for KinFace-II Dataset
# ============================================
# This script trains EG3D on the preprocessed KinFace-II dataset
# with tmux integration for uninterrupted training on remote servers.

set -e  # Exit on error

echo "========================================"
echo "EG3D Training Setup for KinFace-II"
echo "========================================"

# Configuration
DATASET_ZIP="../../KinFaceW-II-EG3D.zip"
PRETRAINED_MODEL="../../eg3d/pretrained/ffhqrebalanced512-128.pkl"
OUTPUT_DIR="../../training-runs"
GPUS=1
BATCH=4
GAMMA=5
KIMG=1000

# Check if dataset exists
if [ ! -f "$DATASET_ZIP" ]; then
    echo "❌ Error: Dataset not found: $DATASET_ZIP"
    echo "Please run prepare_dataset.py first!"
    exit 1
fi

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "❌ Error: Pretrained model not found: $PRETRAINED_MODEL"
    echo "Please run download_pretrained.py first!"
    exit 1
fi

echo "✅ Dataset: $DATASET_ZIP"
echo "✅ Pretrained model: $PRETRAINED_MODEL"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  nvidia-smi not found. Make sure CUDA is installed!"
fi

# Ask for confirmation
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Training command
TRAIN_CMD="python ../../eg3d/eg3d/train.py \
    --outdir=$OUTPUT_DIR \
    --data=$DATASET_ZIP \
    --resume=$PRETRAINED_MODEL \
    --cfg=ffhq \
    --gpus=$GPUS \
    --batch=$BATCH \
    --gamma=$GAMMA \
    --gen_pose_cond=True \
    --gpc_reg_prob=0.7 \
    --neural_rendering_resolution_final=128 \
    --aug=ada \
    --kimg=$KIMG \
    --snap=50 \
    --metrics=fid50k_full"

echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "GPUs: $GPUS"
echo "Batch size: $BATCH"
echo "Gamma (R1 regularization): $GAMMA"
echo "Training images (kimg): $KIMG"
echo "Neural rendering resolution: 64→128"
echo "Augmentation: ADA (adaptive)"
echo "Generator pose conditioning: True"
echo "========================================"
echo ""

# Run without tmux (direct mode)
echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

cd ../../eg3d
eval $TRAIN_CMD
