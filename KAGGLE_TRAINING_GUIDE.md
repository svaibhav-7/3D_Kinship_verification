# Training EG3D on Kaggle - Step-by-Step Guide

## 🎯 Why Kaggle?

Kaggle offers:
- ✅ **Free GPU**: Tesla P100 or T4 (16GB VRAM)
- ✅ **30 hours/week GPU quota** (plenty for 1000 kimg training)
- ✅ **Persistent outputs**: Download trained models
- ✅ **No SSH needed**: All in browser
- ✅ **Easy dataset upload**

---

## 📋 Prerequisites

Before starting, you need:
1. ✅ Kaggle account (free)
2. ✅ GitHub repository with your code
3. ✅ Preprocessed dataset ready

---

## 🚀 Complete Workflow

### **Step 1: Push Code to GitHub**

First, push your code to GitHub:

```bash
# In your local project directory
git remote add origin https://github.com/YOUR_USERNAME/3D-Kinship-Verification.git
git branch -M main
git push -u origin main
```

**What to push:**
- ✅ `project/` directory (all scripts)
- ✅ `README.md`, `PROJECT_OVERVIEW.md`, `SETUP_GUIDE.md`
- ✅ `.gitignore`

**What NOT to push** (already in .gitignore):
- ❌ `KinFaceW-II-Processed/` (upload to Kaggle dataset instead)
- ❌ `env/` (virtual environment)
- ❌ `eg3d/` (will clone in Kaggle)

---

### **Step 2: Create Kaggle Dataset with Preprocessed Images**

1. **Go to Kaggle**: https://www.kaggle.com/datasets
2. **Click "New Dataset"**
3. **Upload settings:**
   - **Title**: `kinface-ii-processed-256`
   - **Files**: Upload entire `KinFaceW-II-Processed/` folder
   - **Visibility**: Private (or Public if allowed)
4. **Click "Create"**

**Alternative - Upload as ZIP:**
```bash
# On your local machine, create ZIP
cd "d:\SasiVaibhav\klu\3rd year\projects\3D_Kinship_Verification"
Compress-Archive -Path KinFaceW-II-Processed -DestinationPath kinface-processed.zip

# Then upload kinface-processed.zip to Kaggle dataset
```

---

### **Step 3: Create Kaggle Notebook**

1. **Go to**: https://www.kaggle.com/code
2. **Click "New Notebook"**
3. **Settings (右侧 sidebar):**
   - **Accelerator**: GPU T4 or P100 ✅
   - **Environment**: Python
   - **Persistence**: Files Only
4. **Add Dataset:**
   - Click "Add Data" → "Your Datasets"
   - Select `kinface-ii-processed-256`

---

### **Step 4: Setup Environment in Kaggle Notebook**

**Cell 1: Clone Repository**
```python
!git clone https://github.com/YOUR_USERNAME/3D-Kinship-Verification.git
%cd 3D-Kinship-Verification
!ls -la
```

**Cell 2: Clone EG3D**
```python
!git clone https://github.com/NVlabs/eg3d.git
%cd eg3d
!git submodule update --init --recursive
%cd ..
```

**Cell 3: Check GPU**
```python
!nvidia-smi
```

**Output should show:** Tesla T4 or P100 with ~16GB memory

---

### **Step 5: Install Dependencies**

**Cell 4: Install EG3D Dependencies**
```python
# Install main dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install ninja imageio imageio-ffmpeg pyspng psutil scipy tqdm click pillow

# Install additional requirements
%cd project/eg3d_setup
!pip install -r requirements_eg3d.txt
%cd ../..
```

---

### **Step 6: Prepare Dataset**

**Cell 5: Link Kaggle Dataset**
```python
import os
import shutil

# Kaggle datasets are mounted at /kaggle/input/
kaggle_dataset_path = "/kaggle/input/kinface-ii-processed-256/KinFaceW-II-Processed"

# Create symlink or copy
if os.path.exists(kaggle_dataset_path):
    !ln -s {kaggle_dataset_path} ./KinFaceW-II-Processed
    print("✅ Dataset linked successfully!")
    !ls -lh KinFaceW-II-Processed/
else:
    print("❌ Dataset not found. Check dataset name in Kaggle!")
```

**Cell 6: Run Camera Extraction**
```python
%cd project/eg3d_setup
!python camera_extraction.py
%cd ../..
```

**Cell 7: Prepare EG3D Dataset**
```python
%cd project/eg3d_setup
!python prepare_dataset.py
%cd ../..
```

---

### **Step 7: Download Pretrained Model**

**Cell 8: Download FFHQ Pretrained Model**
```python
%cd project/eg3d_setup
!python download_pretrained.py
%cd ../..

# Alternative if script fails - manual download
# !mkdir -p eg3d/pretrained
# !wget -O eg3d/pretrained/ffhqrebalanced512-128.pkl \
#   "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-128.pkl"
```

---

### **Step 8: Verify Setup**

**Cell 9: Check Everything**
```python
import os

checks = {
    "Dataset prepared": os.path.exists("KinFaceW-II-EG3D.zip"),
    "Camera params": os.path.exists("project/eg3d_setup/camera_params/camera_params.json"),
    "Pretrained model": os.path.exists("eg3d/pretrained/ffhqrebalanced512-128.pkl"),
    "EG3D repo": os.path.exists("eg3d/eg3d/train.py"),
}

for check, status in checks.items():
    print(f"{'✅' if status else '❌'} {check}")
```

---

### **Step 9: Start Training**

**Cell 10: Training Configuration**
```python
# Training parameters (adjust for Kaggle)
config = {
    "dataset": "./KinFaceW-II-EG3D.zip",
    "pretrained": "./eg3d/pretrained/ffhqrebalanced512-128.pkl",
    "outdir": "./training-runs",
    "gpus": 1,
    "batch": 4,  # Adjust based on GPU memory
    "gamma": 5,
    "kimg": 1000,
}

print("Training Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")
```

**Cell 11: Run Training**
```python
%cd eg3d

# Training command
!python eg3d/train.py \
    --outdir=../training-runs \
    --data=../KinFaceW-II-EG3D.zip \
    --resume=./pretrained/ffhqrebalanced512-128.pkl \
    --cfg=ffhq \
    --gpus=1 \
    --batch=4 \
    --gamma=5 \
    --gen_pose_cond=True \
    --gpc_reg_prob=0.7 \
    --neural_rendering_resolution_final=128 \
    --aug=ada \
    --kimg=1000 \
    --snap=50 \
    --metrics=fid50k_full

%cd ..
```

**Expected output:**
- Initialization messages
- FID calculation
- Progress updates every 50 kimg
- Snapshots saved in `training-runs/`

---

### **Step 10: Monitor Training**

**Cell 12: Check Training Progress**
```python
# Check latest training run
import os
import glob

runs = sorted(glob.glob("training-runs/*"))
if runs:
    latest_run = runs[-1]
    print(f"📁 Latest run: {latest_run}")
    
    # Show FID metrics
    fid_file = os.path.join(latest_run, "metric-fid50k_full.jsonl")
    if os.path.exists(fid_file):
        !tail -5 {fid_file}
    
    # Show training stats
    log_file = os.path.join(latest_run, "log.txt")
    if os.path.exists(log_file):
        !tail -20 {log_file}
else:
    print("No training runs found yet")
```

---

### **Step 11: Test Multi-View Generation**

**Cell 13: Generate Test Views**
```python
import glob

# Find latest snapshot
snapshots = sorted(glob.glob("training-runs/*/network-snapshot-*.pkl"))
if snapshots:
    latest_snapshot = snapshots[-1]
    print(f"Using: {latest_snapshot}")
    
    # Generate 8-view images
    %cd project/eg3d_setup
    !python generate_multiview.py \
        --network ../../{latest_snapshot} \
        --num_views 8 \
        --seeds 0-5 \
        --outdir ../../test_output
    %cd ../..
else:
    print("❌ No snapshots found yet. Wait for training to progress.")
```

**Cell 14: Display Generated Images**
```python
from IPython.display import Image, display
import glob

# Show generated grids
grids = sorted(glob.glob("test_output/*_grid.png"))
for grid in grids[:5]:  # Show first 5
    print(f"\n{grid}")
    display(Image(grid, width=800))
```

---

### **Step 12: Save and Download Results**

**Cell 15: Prepare Downloads**
```python
import shutil

# Create download package
!mkdir -p /kaggle/working/eg3d_results

# Copy trained models
latest_run = sorted(glob.glob("training-runs/*"))[-1]
!cp {latest_run}/network-snapshot-*.pkl /kaggle/working/eg3d_results/

# Copy generated samples
!cp -r test_output /kaggle/working/eg3d_results/

# Create summary
with open("/kaggle/working/eg3d_results/training_summary.txt", "w") as f:
    f.write(f"Training Run: {latest_run}\n")
    f.write(f"Snapshots: {len(glob.glob(f'{latest_run}/network-snapshot-*.pkl'))}\n")

print("✅ Results prepared in /kaggle/working/eg3d_results")
!ls -lh /kaggle/working/eg3d_results/
```

**Download from Kaggle:**
- Click on **Output** tab (right side)
- Download `eg3d_results/` folder
- Contains trained models and sample outputs

---

## ⏱️ Timing Expectations

On Kaggle GPU (T4/P100):

| Checkpoint | Time | Can do in Kaggle? |
|------------|------|-------------------|
| 100 kimg | ~6-8 hours | ✅ Yes (1 session) |
| 500 kimg | ~24-30 hours | ✅ Yes (need to restart) |
| 1000 kimg | ~48-60 hours | ⚠️ Need multiple sessions |

**Kaggle limits:**
- 30 hours/week GPU time
- 12 hours max per session

**Strategy for 1000 kimg:**
1. Train 500 kimg in first session (~24h)
2. Save checkpoint
3. Resume in new session for another 500 kimg

---

## 🔄 Resuming Training (If Session Timeout)

If Kaggle session times out, resume training:

**Cell: Resume Training**
```python
import glob

# Find latest checkpoint
checkpoints = sorted(glob.glob("training-runs/*/network-snapshot-*.pkl"))
latest_checkpoint = checkpoints[-1]
print(f"Resuming from: {latest_checkpoint}")

%cd eg3d

!python eg3d/train.py \
    --outdir=../training-runs \
    --data=../KinFaceW-II-EG3D.zip \
    --resume=../{latest_checkpoint} \
    --cfg=ffhq \
    --gpus=1 \
    --batch=4 \
    --gamma=5 \
    --gen_pose_cond=True \
    --gpc_reg_prob=0.7 \
    --neural_rendering_resolution_final=128 \
    --aug=ada \
    --kimg=1000 \
    --snap=50 \
    --metrics=fid50k_full

%cd ..
```

---

## 📝 Complete Notebook Template

I'll create a ready-to-use Kaggle notebook file for you:

**File**: `kaggle_eg3d_training.ipynb`

Key cells in order:
1. Clone repos
2. Check GPU
3. Install dependencies
4. Link dataset
5. Run setup scripts
6. Download pretrained model
7. Verify setup
8. Start training
9. Monitor progress
10. Generate samples
11. Download results

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size in training command
--batch=2  # Instead of 4
```

### Session Timeout
```python
# Before timeout, download critical files:
!zip -r checkpoint_backup.zip training-runs/
# Download from Output tab
```

### Dataset Not Found
```python
# Check dataset path
!ls /kaggle/input/
# Update path in symlink command
```

---

## ✅ Checklist

Before starting:
- [ ] Code pushed to GitHub
- [ ] KinFace-II-Processed uploaded to Kaggle dataset
- [ ] Kaggle notebook created with GPU
- [ ] Dataset added to notebook

During setup:
- [ ] EG3D cloned
- [ ] Dependencies installed
- [ ] Dataset linked
- [ ] Camera params extracted
- [ ] Pretrained model downloaded

Training:
- [ ] Training started
- [ ] Progress monitored
- [ ] Checkpoints verified
- [ ] Test generation successful

---

## 🎯 Final Notes

**Advantages of Kaggle:**
- Free GPU access
- No tmux needed (browser-based)
- Easy to share with team
- Persistent outputs

**Limitations:**
- 12h session limit (need to resume)
- 30h/week GPU quota
- Slower than RTX 3090

**Recommendation:**
- Train to 500-1000 kimg on Kaggle
- Good enough for research purposes
- Can continue training locally if needed

---

**Ready to start!** Follow the steps in order and you'll have your model training on Kaggle's free GPU! 🚀
