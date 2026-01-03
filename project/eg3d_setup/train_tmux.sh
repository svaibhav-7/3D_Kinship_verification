#!/bin/bash

# EG3D Training with tmux for Uninterrupted Training
# ===================================================
# This script starts EG3D training in a detached tmux session.
# Training will continue even if you disconnect from SSH.
#
# Usage:
#   ./train_tmux.sh          # Start new training session
#   tmux attach -t eg3d      # Reconnect to training session
#   tmux kill-session -t eg3d  # Stop training

set -e

SESSION_NAME="eg3d_kinface"

echo "========================================"
echo "EG3D Training with tmux"
echo "========================================"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ Error: tmux is not installed!"
    echo ""
    echo "Install tmux:"
    echo "  Ubuntu/Debian: sudo apt-get install tmux"
    echo "  CentOS/RHEL:   sudo yum install tmux"
    echo "  macOS:         brew install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Training session '$SESSION_NAME' is already running!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill and restart:           tmux kill-session -t $SESSION_NAME && $0"
    echo ""
    exit 1
fi

# Configuration
DATASET_ZIP="$(pwd)/../../KinFaceW-II-EG3D.zip"
PRETRAINED_MODEL="$(pwd)/../../eg3d/pretrained/ffhqrebalanced512-128.pkl"
OUTPUT_DIR="$(pwd)/../../training-runs"
GPUS=1
BATCH=4
GAMMA=5
KIMG=1000

# Verify prerequisites
echo "Checking prerequisites..."

if [ ! -f "$DATASET_ZIP" ]; then
    echo "❌ Error: Dataset not found: $DATASET_ZIP"
    exit 1
fi
echo "✅ Dataset: $DATASET_ZIP"

if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "❌ Error: Pretrained model not found: $PRETRAINED_MODEL"
    exit 1
fi
echo "✅ Pretrained model: $PRETRAINED_MODEL"

# Check GPU
echo ""
echo "GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found"
fi

echo ""
echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "Session name: $SESSION_NAME"
echo "GPUs: $GPUS"
echo "Batch size: $BATCH"
echo "Gamma: $GAMMA"
echo "Training kimg: $KIMG"
echo "========================================"
echo ""

# Confirmation
read -p "Start training in tmux session? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create training script in tmux
echo "Creating tmux session '$SESSION_NAME'..."

tmux new-session -d -s $SESSION_NAME

# Configure tmux session
tmux send-keys -t $SESSION_NAME "cd $(pwd)/../../eg3d" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'EG3D Training Started'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Time: \$(date)'" C-m
tmux send-keys -t $SESSION_NAME "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m

# Activate virtual environment if needed
if [ -d "../../env" ]; then
    tmux send-keys -t $SESSION_NAME "source ../../env/bin/activate" C-m
fi

# Start training
TRAIN_CMD="python eg3d/train.py \
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

tmux send-keys -t $SESSION_NAME "$TRAIN_CMD" C-m

echo ""
echo "========================================" 
echo "✅ Training started in tmux session!"
echo "========================================"
echo ""
echo "Useful commands:"
echo "  View training:      tmux attach -t $SESSION_NAME"
echo "  Detach (keep running): Press Ctrl+B, then D"
echo "  Stop training:      tmux kill-session -t $SESSION_NAME"
echo "  List sessions:      tmux ls"
echo ""
echo "Training logs will be in: $OUTPUT_DIR"
echo ""
echo "To reconnect after SSH disconnect:"
echo "  ssh your_server"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "========================================"
