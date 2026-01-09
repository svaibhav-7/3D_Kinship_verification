# EG3D Training - Universal Script

## 🌍 Works Everywhere!

This single script works in **both** Local and Kaggle environments!

### ✨ Features

- ✅ **Auto-detection**: Automatically detects Kaggle vs Local
- ✅ **User choice**: Select environment (1=Local, 2=Kaggle)
- ✅ **Platform support**: Windows, Linux, macOS
- ✅ **Robust error handling**: Clear error messages
- ✅ **Path resolution**: Automatically finds datasets
- ✅ **Complete automation**: All 6 steps in one run

---

## 🚀 Usage

### In Kaggle Notebook:

```python
!wget https://raw.githubusercontent.com/svaibhav-7/3D_Kinship_verification/main/train_eg3d_universal.py
%run train_eg3d_universal.py
```

**Output:**
```
🔍 Detected: Kaggle Notebook
🔔 Running in Kaggle - Using Kaggle configuration
✅ Selected: KAGGLE Training
```

---

### On Local Machine:

```bash
# Clone repository
git clone https://github.com/svaibhav-7/3D_Kinship_verification.git
cd 3D_Kinship_verification

# Run script
python train_eg3d_universal.py
```

**Output:**
```
🔍 Detected: Local Machine (Windows/Linux)
Select training environment:
  1. Local Training
  2. Kaggle Training
Enter choice (1 or 2): 1
✅ Selected: LOCAL Training
```

---

## 📋 What It Does

### Step-by-Step:

1. **Environment Detection**
   - Detects Kaggle vs Local automatically
   - Asks for user confirmation
   - Configures paths accordingly

2. **Dataset Verification**
   - Finds KinFace-II-Processed dataset
   - Counts images (should be 2000)
   - Links Kaggle dataset (if in Kaggle)

3. **Camera Extraction**
   - Extracts camera parameters
   - Creates camera_params.json
   - Takes ~5 seconds for 2000 images

4. **Dataset Preparation**
   - Creates EG3D-compatible format
   - Mirrors images (doubles to 4000)
   - Generates ZIP (~500 MB)

5. **Download Model**
   - Downloads ffhqrebalanced512-128.pkl
   - ~650 MB from NVIDIA NGC
   - Skips if already exists

6. **Start Training**
   - Runs EG3D training
   - 1000 kimg target
   - Takes 24-48 hours on GPU

---

## ⚙️ Configuration

### Automatic Configuration:

**Kaggle:**
- Dataset: `/kaggle/input/kinface-ii-processed-256/`
- GPU: Tesla T4 (free)
- Batch: 4
- Output: `training-runs/`

**Local:**
- Dataset: `./KinFaceW-II-Processed/`
- GPU: Your GPU (CUDA required)
- Batch: 4 (adjustable)
- Output: `./training-runs/`

---

## 🔧 Requirements

### Kaggle (Auto-installed):
- Python 3.8+
- PyTorch (pre-installed)
- opencv-python (auto-installed)
- numpy, pillow, tqdm

### Local:
```bash
# Install dependencies
pip install torch torchvision opencv-python numpy pillow tqdm

# Clone EG3D
git clone https://github.com/NVlabs/eg3d.git
cd eg3d && git submodule update --init --recursive
```

---

## 🐛 Troubleshooting

### Error: Dataset not found (Local)
```
❌ ERROR: Dataset not found!
   Expected: ./KinFaceW-II-Processed/
```

**Solution:** Make sure you've run preprocessing:
```bash
cd project/preprocessing
python preprocess_kinface.py
```

---

### Error: EG3D train.py not found
```
❌ ERROR: eg3d/eg3d/train.py not found!
```

**Solution:** Clone EG3D:
```bash
git clone https://github.com/NVlabs/eg3d.git
```

---

### Error: Out of memory (Local)
**Solution:** Reduce batch size by editing script:
```python
"batch": 2,  # Instead of 4
```

---

## 📊 Expected Timeline

| Step | Kaggle | Local (RTX 3090) |
|------|--------|------------------|
| Dataset verification | 10s | 10s |
| Camera extraction | 5s | 5s |
| Dataset prep + ZIP | 2-3 min | 2-3 min |
| Download model | 3-5 min | 3-5 min |
| Verification | 5s | 5s |
| **Training (1000 kimg)** | **24-48h** | **12-24h** |

---

## ✅ Advantages

### vs. Separate Scripts:
- ✅ One script for both environments
- ✅ No manual configuration needed
- ✅ Automatic path resolution
- ✅ Better error messages

### vs. Manual Setup:
- ✅ Fully automated (no manual steps)
- ✅ Validates everything before training
- ✅ Resume-friendly
- ✅ Platform-independent

---

## 🎯 Quick Start Comparison

### OLD WAY (Multiple scripts):
```python
# Cell 1: Clone repos
# Cell 2: Install deps
# Cell 3: Link dataset
# Cell 4: Camera extraction
# Cell 5: Dataset prep
# Cell 6: Download model
# Cell 7: Verify
# Cell 8: Train
# = 8 cells, lots of manual work
```

### NEW WAY (Universal script):
```python
%run train_eg3d_universal.py
# = 1 command, fully automated ✨
```

---

## 📝 Notes

- Environment auto-detected but user can override
- Works on Windows, Linux, macOS
- Kaggle gets free GPU automatically
- Local requires CUDA-capable GPU
- All paths automatically resolved

---

**Ready to train anywhere!** 🚀

One script, two environments, zero hassle!
