# Kaggle Notebook - Direct Cell Code
# Copy and paste each cell into your Kaggle notebook

## Cell 1: Clone Repositories
```python
!git clone https://github.com/svaibhav-7/3D_Kinship_verification.git
%cd 3D_Kinship_verification
!git clone https://github.com/NVlabs/eg3d.git
%cd eg3d && !git submodule update --init --recursive && %cd ..
!nvidia-smi
```

## Cell 2: Install Dependencies
```python
!pip install torch torchvision ninja imageio pyspng scipy tqdm click pillow opencv-python
%cd project/eg3d_setup
!cat requirements_eg3d.txt
!pip install -r requirements_eg3d.txt
%cd ../..
```

## Cell 3: Link Kaggle Dataset
```python
import os
!ln -s /kaggle/input/kinface-ii-processed-256/KinFaceW-II-Processed ./KinFaceW-II-Processed
!ls -lh KinFaceW-II-Processed/ | head -20
```

## Cell 4: Camera Extraction (Kaggle Version)
```python
# Run camera extraction
%run project/eg3d_setup/camera_extraction_kaggle.py
```

## Cell 5: Dataset Preparation (Kaggle Version)
```python
# Run dataset preparation  
%run project/eg3d_setup/prepare_dataset_kaggle.py
```

## Cell 6: Download Pretrained Model
```python
!mkdir -p eg3d/pretrained

# Download FFHQ pretrained model
!wget -O eg3d/pretrained/ffhqrebalanced512-128.pkl \
  "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-128.pkl"

# Check file size (should be ~650 MB)
!ls -lh eg3d/pretrained/
```

## Cell 7: Verify Setup
```python
import os

checks = {
    "Dataset ZIP": os.path.exists("KinFaceW-II-EG3D.zip"),
    "Camera params": os.path.exists("camera_params/camera_params.json"),
    "Pretrained model": os.path.exists("eg3d/pretrained/ffhqrebalanced512-128.pkl"),
    "Dataset.json": os.path.exists("KinFaceW-II-EG3D/dataset.json"),
}

print("\n" + "="*50)
print("Setup Verification")
print("="*50)
for check, status in checks.items():
    print(f"{'✅' if status else '❌'} {check}")
print("="*50)

if all(checks.values()):
    print("\n🎉 All checks passed! Ready to train!")
else:
    print("\n⚠️ Fix failed checks before training")
```

## Cell 8: START TRAINING
```python
%cd eg3d

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

## Cell 9: Monitor Training Progress
```python
import glob

# Find latest training run
runs = sorted(glob.glob("training-runs/*"))
if runs:
    latest_run = runs[-1]
    print(f"📁 Latest run: {latest_run}\n")
    
    # Show training log
    log_file = f"{latest_run}/log.txt"
    if os.path.exists(log_file):
        print("📊 Training Log (last 30 lines):")
        !tail -30 {log_file}
    
    # Show FID metrics
    fid_file = f"{latest_run}/metric-fid50k_full.jsonl"
    if os.path.exists(fid_file):
        print("\n📈 FID Metrics:")
        !tail -5 {fid_file}
    
    # List snapshots
    snapshots = sorted(glob.glob(f"{latest_run}/network-snapshot-*.pkl"))
    print(f"\n💾 Snapshots: {len(snapshots)}")
    for snap in snapshots[-5:]:
        print(f"  {os.path.basename(snap)}")
else:
    print("No training runs found yet")
```

## Cell 10: Generate Test Multi-View Images
```python
import glob
from pathlib import Path

# Find latest snapshot
snapshots = sorted(glob.glob("training-runs/*/network-snapshot-*.pkl"))

if snapshots:
    latest_snapshot = snapshots[-1]
    print(f"Using: {latest_snapshot}\n")
    
    # Generate 8-view images
    %cd project/eg3d_setup
    !python generate_multiview.py \
        --network ../../{latest_snapshot} \
        --num_views 8 \
        --seeds 0-5 \
        --outdir ../../test_output
    %cd ../..
    
    print("\n✅ Multi-view images generated!")
    print("Check test_output/ folder")
else:
    print("❌ No snapshots found. Wait for training to progress (at least 50 kimg)")
```

## Cell 11: Display Generated Images
```python
from IPython.display import Image, display
import glob

# Show generated image grids
grids = sorted(glob.glob("test_output/*_grid.png"))

print(f"Found {len(grids)} image grids\n")

for grid in grids[:10]:  # Show first 10
    print(f"📷 {grid}")
    display(Image(grid, width=800))
    print("\n")
```

## Cell 12: Prepare Results for Download
```python
import shutil

# Create download package
output_dir = "/kaggle/working/eg3d_results"
!mkdir -p {output_dir}

# Find latest training run
latest_run = sorted(glob.glob("training-runs/*"))[-1]

# Copy trained models (latest 3 snapshots)
snapshots = sorted(glob.glob(f"{latest_run}/network-snapshot-*.pkl"))[-3:]
for snap in snapshots:
    shutil.copy(snap, output_dir)
    print(f"Copied: {os.path.basename(snap)}")

# Copy test outputs
if os.path.exists("test_output"):
    shutil.copytree("test_output", f"{output_dir}/samples", dirs_exist_ok=True)
    print("Copied: test output samples")

# Copy training logs
for log_file in ["log.txt", "metric-fid50k_full.jsonl"]:
    src = f"{latest_run}/{log_file}"
    if os.path.exists(src):
        shutil.copy(src, output_dir)
        print(f"Copied: {log_file}")

# Summary
print("\n" + "="*50)
print("✅ Results ready for download!")
print("="*50)
!ls -lh {output_dir}
print("\n📥 Download from: Output tab → eg3d_results/")
```

---

## 📝 Notes

- **Run cells in order** from top to bottom
- **Training time**: ~24-48 hours for 1000 kimg on T4/P100
- **GPU quota**: Kaggle gives 30 hours/week
- **Session limit**: 12 hours max, need to resume if timeout

## ⚠️ If Session Times Out

Resume training by running Cell 8 again, but change `--resume` to point to your latest snapshot:

```python
# Find latest checkpoint
latest_checkpoint = sorted(glob.glob("training-runs/*/network-snapshot-*.pkl"))[-1]
print(f"Resuming from: {latest_checkpoint}")

# Then update Cell 8 with:
# --resume=../{latest_checkpoint}
```
