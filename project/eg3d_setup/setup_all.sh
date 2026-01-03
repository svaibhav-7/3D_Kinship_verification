#!/bin/bash

# Complete EG3D Setup Pipeline
# ============================
# Run all setup steps in sequence

set -e

echo "========================================"
echo "EG3D Complete Setup Pipeline"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Camera extraction
echo "Step 1/3: Extracting camera parameters..."
python camera_extraction.py
if [ $? -ne 0 ]; then
    echo "❌ Camera extraction failed!"
    exit 1
fi
echo "✅ Camera extraction complete"
echo ""

# Step 2: Dataset preparation
echo "Step 2/3: Preparing EG3D dataset..."
python prepare_dataset.py
if [ $? -ne 0 ]; then
    echo "❌ Dataset preparation failed!"
    exit 1
fi
echo "✅ Dataset preparation complete"
echo ""

# Step 3: Download pretrained model
echo "Step 3/3: Downloading pretrained model..."
python download_pretrained.py
if [ $? -ne 0 ]; then
    echo "❌ Model download failed!"
    exit 1
fi
echo "✅ Pretrained model downloaded"
echo ""

# Summary
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Start training: ./train_tmux.sh"
echo "  2. Monitor: tmux attach -t eg3d_kinface"
echo "  3. Generate views: python generate_multiview.py --help"
echo ""
echo "========================================"
