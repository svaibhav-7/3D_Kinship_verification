# EG3D Training Setup - Complete Guide

## 🎯 Overview

This directory contains all scripts needed to train EG3D on the preprocessed KinFace-II dataset for generating 8-view 3D-consistent face images.

**What's included:**
- Camera parameter extraction
- EG3D dataset preparation
- Pretrained model download
- Training scripts with tmux integration
- Multi-view image generation

---

## 📋 Prerequisites

### Hardware
- High-end NVIDIA GPU (RTX 3090, V100, A100 recommended)
- 16GB+ GPU RAM
- 32GB+ system RAM
- 100GB+ free disk space

### Software
- Linux (Ubuntu 18.04+recommended)
- Python 3.8+
- CUDA 11.3+
- tmux (for uninterrupted training)

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Install EG3D dependencies
cd ../../eg3d
pip install -r eg3d/environment.yml

# OR if using conda (recommended)
conda env create -f eg3d/environment.yml
conda activate eg3d

# Install tmux
sudo apt-get install tmux  # Ubuntu/Debian
# OR
sudo yum install tmux  # CentOS/RHEL
```

### Step 2: Extract Camera Parameters

```bash
cd project/eg3d_setup
python camera_extraction.py
```

**Output:** `camera_params/camera_params.json` with camera parameters for all images.

> **Note**: Currently uses default front-facing camera parameters. For better results, update the script to use Deep3DFaceRecon_pytorch.

### Step 3: Prepare EG3D Dataset

```bash
python prepare_dataset.py
```

**Output:**
- `KinFaceW-II-EG3D/` directory with dataset.json
- `KinFaceW-II-EG3D.zip` uncompressed archive (~2000 images × 2 with mirroring)

### Step 4: Download Pretrained Model

```bash
python download_pretrained.py
```

**Downloads:** `ffhqrebalanced512-128.pkl` (~650 MB) to `eg3d/pretrained/`

**Manual download** (if script fails):
1. Visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d
2. Download `ffhqrebalanced512-128.pkl`
3. Place in `eg3d/pretrained/`

### Step 5: Start Training

#### Option A: With tmux (Recommended for Remote Servers)

```bash
chmod +x train_tmux.sh
./train_tmux.sh
```

**Benefits:**
- Training continues if SSH disconnects
- Reconnect anytime with: `tmux attach -t eg3d_kinface`
- Detach without stopping: `Ctrl+B`, then `D`

#### Option B: Direct Training

```bash
chmod +x train.sh
./train.sh
```

**Training time:**
- ~2-3 days on RTX 3090 (1000 kimg)
- ~5-7 days on GTX 1080 Ti
- Check progress in `training-runs/`

### Step 6: Generate Multi-View Images

After training (or using preting model):

```bash
python generate_multiview.py \
    --network ../../training-runs/latest/network-snapshot-001000.pkl \
    --num_views 8 \
    --seeds 0-10 \
    --outdir ./output
```

**Output:** 8 views per face at angles: -60°, -45°, -30°, -15°, 0°, +15°, +30°, +45°, +60°

---

## 📁 File Structure

```
eg3d_setup/
├── camera_extraction.py       # Extract camera params from images
├── prepare_dataset.py          # Create EG3D-compatible dataset
├── download_pretrained.py      # Download pretrained FFHQ model
├── train.sh                    # Direct training script
├── train_tmux.sh              # Training with tmux integration ⭐
├── generate_multiview.py      # Generate 8-view images
├── requirements_eg3d.txt      # Additional dependencies
└── README.md                   # This file

Generated files:
├── camera_params/
│   └── camera_params.json     # Camera parameters for all images
└── multiview_output/           # Generated multi-view images
```

---

## ⚙️ Training Configuration

**Default parameters** (`train_tmux.sh`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--gpus` | 1 | Number of GPUs |
| `--batch` | 4 | Batch size (adjust for GPU memory) |
| `--gamma` | 5 | R1 regularization strength |
| `--kimg` | 1000 | Training images in thousands |
| `--gen_pose_cond` | True | Generator pose conditioning |
| `--aug` | ada | Adaptive data augmentation |
| `--neural_rendering_resolution_final` | 128 | Final rendering resolution |

**Adjust for your GPU:**

- **RTX 3090 (24GB)**: batch=8, gpus=1
- **RTX 3080 (10GB)**: batch=4, gpus=1
- **V100 (32GB)**: batch=8-16, gpus=1
- **Multiple GPUs**: batch=32, gpus=4-8

Edit values in `train_tmux.sh` before running.

---

## 🔧 Tmux Commands

### Essential Commands

```bash
# Start training
./train_tmux.sh

# View training (attach to session)
tmux attach -t eg3d_kinface

# Detach (keep training running)
# Press: Ctrl+B, then D

# List all sessions
tmux ls

# Kill training session
tmux kill-session -t eg3d_kinface
```

### Advanced Tmux

```bash
# Split panes to monitor GPU while training
tmux attach -t eg3d_kinface
# Press: Ctrl+B, then "  (vertical split)
# In new pane: watch -n 1 nvidia-smi

# Scroll in tmux
# Press: Ctrl+B, then [
# Use arrow keys, then q to exit

# Create custom session name
SESSION_NAME="my_experiment" ./train_tmux.sh
```

---

## 📊 Monitoring Training

### Training Logs

```bash
# View latest logs
tail -f ../../training-runs/00000-ffhq-kinface/log.txt

# View FID metrics
cat ../../training-runs/00000-ffhq-kinface/metric-fid50k_full.jsonl
```

### TensorBoard (if installed)

```bash
tensorboard --logdir ../../training-runs
# Open: http://localhost:6006
```

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# In separate tmux pane
tmux split-window -v
watch -n 1 nvidia-smi
```

---

## 🎨 Multi-View Generation Examples

### Generate from Trained Model

```bash
# Single seed, 8 views
python generate_multiview.py \
    --network ../../training-runs/latest/network-snapshot-001000.pkl \
    --num_views 8 \
    --seeds 0 \
    --outdir ./demo_output

# Multiple seeds for comparison
python generate_multiview.py \
    --network ../../training-runs/latest/network-snapshot-001000.pkl \
    --num_views 8 \
    --seeds 0-20 \
    --outdir ./batch_output
```

### Use Pretrained Model (No Training)

```bash
python generate_multiview.py \
    --network ../../eg3d/pretrained/ffhqrebalanced512-128.pkl \
    --num_views 8 \
    --seeds 0-10 \
    --outdir ./pretrained_output
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
# Edit train_tmux.sh: BATCH=2
```

### Tmux Session Not Found

```bash
# Check if session exists
tmux ls

# Create new session
./train_tmux.sh
```

### Training Not Starting

```bash
# Check dataset exists
ls -lh ../../KinFaceW-II-EG3D.zip

# Check pretrained model
ls -lh ../../eg3d/pretrained/

# Verify GPU
nvidia-smi
```

### Deep3DFaceRecon Not Found

```bash
# Initialize submodule
cd ../../eg3d
git submodule update --init --recursive
```

---

## 📈 Expected Results

### Training Progress

- **100 kimg**: Basic face structure, low diversity
- **500 kimg**: Good faces, improving 3D consistency
- **1000 kimg**: High-quality faces, good multi-view consistency ✅
- **2000 kimg**: Best quality (optional)

### FID Scores

- **Baseline (pretrained FFHQ)**: FID ~10-15
- **After 500 kimg**: FID ~20-30
- **After 1000 kimg**: FID ~15-25 (target)

---

## 🔄 Next Steps After Training

1. **Evaluate multi-view consistency**
   ```bash
   python generate_multiview.py --num_views 16 --seeds 0-100
   ```

2. **Generate 3D kinship verification dataset**
   - Generate 8 views for all KinFace-II images
   - Use for 3D-aware kinship verification

3. **Fine-tune further** (optional)
   - Increase `--kimg` to 2000
   - Adjust `--gamma` for better quality

---

## 📝 Notes for Team Members

### Before Training

1. ✅ GPU with 16GB+ VRAM available?
2. ✅ CUDA 11.3+ installed?
3. ✅ Enough disk space (~100GB)?
4. ✅ tmux installed for remote training?

### During Training

- Training will take 2-3 days
- Check logs every few hours
- Monitor GPU temperature (<85°C)
- FID metrics saved every 50 kimg

### After Training

- Model snapshots in `training-runs/`
- Use latest snapshot for inference
- Compare with pretrained model quality

---

## 📚 Additional Resources

- **EG3D Paper**: https://nvlabs.github.io/eg3d/
- **GitHub**: https://github.com/NVlabs/eg3d
- **Pretrained Models**: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d
- **Deep3DFaceRecon**: https://github.com/sicxu/Deep3DFaceRecon_pytorch

---

## ✅ Checklist

- [ ] Install dependencies (EG3D, tmux)
- [ ] Run `camera_extraction.py`
- [ ] Run `prepare_dataset.py`
- [ ] Run `download_pretrained.py`
- [ ] Start training with `train_tmux.sh`
- [ ] Monitor training progress
- [ ] Generate multi-view images
- [ ] Evaluate results

---

**Questions?** Check the main project README or implementation_plan.md in the brain directory.

**Ready to train!** 🚀
