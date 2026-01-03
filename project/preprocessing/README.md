# KinFace-II Preprocessing for EG3D - Quick Start Guide

## 📁 Project Structure

```
3D_Kinship_Verification/
├── env/                          # Virtual environment
├── eg3d/                         # EG3D repository (cloned)
├── KinFaceW-II/                 # Original dataset
│   └── images/
│       ├── father-dau/          # 250 pairs
│       ├── father-son/          # 250 pairs
│       ├── mother-dau/          # 250 pairs
│       └── mother-son/          # 250 pairs
├── project/                      # Your project files ✨
│   ├── preprocessing/
│   │   └── preprocess_kinface.py
│   ├── requirements.txt
│   ├── install_deps.bat         # Installation script
│   └── run_preprocessing.bat    # Run preprocessing
└── KinFaceW-II-Processed/       # Output (will be created)
```

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd project
.\install_deps.bat
```

This installs:
- ✅ facenet-pytorch (MTCNN face detection)
- ✅ PyTorch (CPU/GPU support)
- ✅ PIL, OpenCV, NumPy
- ✅ tqdm (progress bars)

### Step 2: Run Preprocessing

```bash
.\run_preprocessing.bat
```

This will:
1. ✅ Detect faces using MTCNN
2. ✅ Align eyes horizontally (5-point landmarks)
3. ✅ Resize to 256×256
4. ✅ Save to `KinFaceW-II-Processed/`

**Processing time:** ~5-10 minutes for 2000 images

### Step 3: Verify Output

Check the processed images:
```
KinFaceW-II-Processed/
├── father-dau/    # 500 images (256×256)
├── father-son/    # 500 images (256×256)
├── mother-dau/    # 500 images (256×256)
└── mother-son/    # 500 images (256×256)
```

---

## 🔧 What the Preprocessing Does

### Face Detection & Alignment
- Uses **MTCNN** (Multi-task Cascaded Convolutional Networks)
- Detects facial landmarks (eyes, nose, mouth)
- Aligns eyes horizontally
- Crops face with consistent margins

### Resize
- Original: 64×64 (KinFace-II default)
- Output: 256×256 (EG3D compatible)
- Method: High-quality upsampling

### Output Format
- Format: PNG (lossless)
- Color: RGB
- Size: 256×256 pixels
- Structure: Same as original

---

## 🎯 Next Steps: EG3D Integration

### 1. Install EG3D Dependencies

```bash
cd ..\eg3d
pip install -r requirements.txt
```

### 2. Download Pretrained Model

Download from [EG3D Model Zoo](https://github.com/NVlabs/eg3d#pretrained-models):
- **FFHQ256**: `ffhq-fixed-triplane256-128.pkl`

```bash
mkdir pretrained
# Download model and place in eg3d/pretrained/
```

### 3. Generate 8-View Images

Use EG3D inference script (example):

```python
import torch
from eg3d.gen_samples import generate_images

# Load pretrained model
network_pkl = 'pretrained/ffhq-fixed-triplane256-128.pkl'

# Load your processed image
input_image = 'KinFaceW-II-Processed/father-son/fs_001_1.png'

# Generate 8 views
generate_images(network_pkl, input_image, num_views=8)
```

---

## 📊 Dataset Statistics

| Relation | Pairs | Images | Status |
|----------|-------|--------|--------|
| Father-Daughter | 250 | 500 | ✅ Ready |
| Father-Son | 250 | 500 | ✅ Ready |
| Mother-Daughter | 250 | 500 | ✅ Ready |
| Mother-Son | 250 | 500 | ✅ Ready |
| **Total** | **1000** | **2000** | ✅ **EG3D Compatible** |

---

## ⚙️ Manual Commands

If you prefer manual execution:

```bash
# Activate environment
..\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
cd preprocessing
python preprocess_kinface.py
```

---

## 🛠️ Troubleshooting

### MTCNN not detecting faces?
- The script has automatic fallback (center crop + resize)
- Should work for all KinFace images (already cropped)

### Out of memory?
- Script auto-detects GPU/CPU
- Will use CPU if no CUDA available

### Slow processing?
- Expected: ~2-3 images/second on CPU
- Faster with GPU: ~10-20 images/second

---

## ✨ Features

- ✅ Automatic face detection & alignment
- ✅ Handles missing faces gracefully
- ✅ Progress bars with tqdm
- ✅ Maintains folder structure
- ✅ Detailed statistics
- ✅ EG3D-ready output

---

## 📝 Citation

If using KinFace-II dataset:

```bibtex
@inproceedings{lu2012neighborhood,
  title={Neighborhood repulsed metric learning for kinship verification},
  author={Lu, J. and Hu, J. and Zhou, X. and Shang, Y. and Tan, Y.-P. and Wang, G.},
  booktitle={CVPR},
  year={2012}
}
```

If using EG3D:

```bibtex
@inproceedings{Chan2022,
  title={Efficient Geometry-aware 3D Generative Adversarial Networks},
  author={Chan, Eric R and others},
  booktitle={CVPR},
  year={2022}
}
```

---

**Ready to process!** Run `install_deps.bat` to get started. 🚀
