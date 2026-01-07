# Quick Start: Kaggle Training (TL;DR)

## 🚀 In Order - What to Do

### 1. **Push to GitHub** (5 minutes)
```bash
git remote add origin https://github.com/YOUR_USERNAME/3D-Kinship-Verification.git
git push -u origin main
```

### 2. **Upload Dataset to Kaggle** (10 minutes)
- Go to kaggle.com/datasets → "New Dataset"
- Upload `KinFaceW-II-Processed/` folder
- Name it: `kinface-ii-processed-256`
- Set to Private

### 3. **Create Kaggle Notebook** (2 minutes)
- kaggle.com/code → "New Notebook"
- **Settings**: Accelerator = **GPU T4**
- **Add Data**: Your dataset (`kinface-ii-processed-256`)

### 4. **Run These Cells in Order:**

**🔹 Cell 1: Clone & Setup**
```python
!git clone https://github.com/YOUR_USERNAME/3D-Kinship-Verification.git
%cd 3D-Kinship-Verification
!git clone https://github.com/NVlabs/eg3d.git
%cd eg3d && !git submodule update --init --recursive && %cd ..
!nvidia-smi
```

**🔹 Cell 2: Install Dependencies**
```python
!pip install torch torchvision ninja imageio pyspng scipy tqdm click pillow
%cd project/eg3d_setup && !pip install -r requirements_eg3d.txt && %cd ../..
```

**🔹 Cell 3: Link Dataset**
```python
!ln -s /kaggle/input/kinface-ii-processed-256/KinFaceW-II-Processed ./KinFaceW-II-Processed
!ls -lh KinFaceW-II-Processed/
```

**🔹 Cell 4: Prepare for Training**
```python
%cd project/eg3d_setup
!python camera_extraction.py
!python prepare_dataset.py
!python download_pretrained.py
%cd ../..
```

**🔹 Cell 5: Verify Setup**
```python
import os
print("✅ Dataset:", os.path.exists("KinFaceW-II-EG3D.zip"))
print("✅ Model:", os.path.exists("eg3d/pretrained/ffhqrebalanced512-128.pkl"))
```

**🔹 Cell 6: START TRAINING** ⭐
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
    --snap=50
```

**Expected Time:** 24-48 hours (may need to resume if session times out)

**🔹 Cell 7: Monitor Progress**
```python
!tail -20 ../training-runs/*/log.txt
```

**🔹 Cell 8: Generate Test Views**
```python
%cd ../project/eg3d_setup
!python generate_multiview.py \
    --network ../../training-runs/*/network-snapshot-*.pkl \
    --num_views 8 \
    --seeds 0-5 \
    --outdir ../../test_output
```

**🔹 Cell 9: Download Results**
```python
# Results are in /kaggle/working/
# Download from Output tab → eg3d_results/
!cp -r ../training-runs /kaggle/working/
!cp -r ../test_output /kaggle/working/
```

---

## ⏱️ Timeline

| Step | Time |
|------|------|
| Setup (cells 1-5) | ~30 min |
| Training 500 kimg | ~24 hours |
| Training 1000 kimg | ~48 hours |
| Download results | ~5 min |

---

## 💡 Key Tips

1. **Kaggle Limits**: 12h per session, 30h/week GPU
2. **Resume Training**: Use `--resume=path/to/snapshot.pkl` if session timeout
3. **Batch Size**: Reduce to `--batch=2` if out of memory
4. **Monitor**: Check Output tab for checkpoints every ~2 hours

---

## 📥 What You'll Download

After training:
- `network-snapshot-XXXXX.pkl` (trained models)
- `test_output/` (8-view sample images)
- `log.txt` (training logs)
- `metric-fid50k_full.jsonl` (quality metrics)

---

**Full detailed guide**: See `KAGGLE_TRAINING_GUIDE.md`

**Questions?** Each cell has comments. Read the full guide for troubleshooting!
