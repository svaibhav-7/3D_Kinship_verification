# 3D Kinship Verification with EG3D

![Status](https://img.shields.io/badge/status-ready_for_training-green)
![Dataset](https://img.shields.io/badge/dataset-KinFaceII-blue)
![Model](https://img.shields.io/badge/model-EG3D-purple)

## 🎯 Project Overview

This project uses pretrained **EG3D GAN** model to generate **8-view 3D-consistent face images** from the KinFace-II dataset for enhanced kinship verification.

### Key Features
- ✅ **Complete preprocessing pipeline** with MTCNN face detection
- ✅ **EG3D training setup** with tmux integration for uninterrupted training
- ✅ **8-view generation** from single face images
- ✅ **Camera parameter extraction** for EG3D compatibility
- ✅ **Mirror augmentation** to double dataset size
- ✅ **Preprocessed dataset included** (256×256 aligned faces)

---

## 🚀 Quick Start

### For Preprocessing (Already Complete)
```bash
cd project/preprocessing
.\install_deps.bat
.\run_preprocessing.bat
```

### For Training (On GPU Server)
```bash
cd project/eg3d_setup

# One-command setup
chmod +x setup_all.sh
./setup_all.sh

# Start training with tmux
chmod +x train_tmux.sh
./train_tmux.sh

# Disconnect SSH - training continues!
```

---

## 📁 Project Structure

```
3D_Kinship_Verification/
├── KinFaceW-II-Processed/          # ✅ Preprocessed dataset (256×256)
├── project/
│   ├── preprocessing/              # ✅ Face detection & alignment
│   │   ├── preprocess_kinface.py
│   │   ├── install_deps.bat
│   │   └── run_preprocessing.bat
│   └── eg3d_setup/                 # ✅ Training pipeline
│       ├── camera_extraction.py    # Extract camera params
│       ├── prepare_dataset.py      # Create EG3D dataset
│       ├── download_pretrained.py  # Download FFHQ model
│       ├── train_tmux.sh          # ⭐ Main training script
│       ├── generate_multiview.py   # Generate 8 views
│       ├── setup_all.sh           # Complete automation
│       └── README.md              # Full documentation
├── PROJECT_OVERVIEW.md             # This file
└── SETUP_GUIDE.md                  # Installation guide
```

---

## 📊 Dataset

**KinFace-II Preprocessed**
- 4 kinship relations: father-dau, father-son, mother-dau, mother-son
- ~2,000 images total
- 256×256 pixels
- Face-aligned with MTCNN (horizontal eye alignment)
- Ready for EG3D training

---

## ⚙️ Requirements

### Preprocessing (Windows)
- Python 3.8+
- facenet-pytorch
- PyTorch (CPU)
- PIL, OpenCV, NumPy

### Training (Linux GPU Server)
- NVIDIA GPU (16GB+ VRAM)
- CUDA 11.3+
- PyTorch 1.11+
- tmux
- See `project/eg3d_setup/requirements_eg3d.txt`

---

## 📖 Documentation

- **Main Guide**: [project/eg3d_setup/README.md](project/eg3d_setup/README.md)
- **Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Project Overview**: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

---

## 🎯 Workflow

1. **Preprocessing** (Complete ✅)
   - MTCNN face detection
   - Eye alignment
   - Resize to 256×256

2. **Dataset Preparation**
   - Extract camera parameters
   - Create EG3D format
   - Mirror augmentation

3. **Training** (2-3 days on RTX 3090)
   - Fine-tune from FFHQ pretrained
   - 1000 kimg recommended
   - Tmux for uninterrupted training

4. **Multi-View Generation**
   - Generate 8 views per face
   - 3D-consistent outputs
   - Ready for kinship verification

---

## 🔑 Key Scripts

### `train_tmux.sh` ⭐
Training script with tmux integration for remote servers:
```bash
./train_tmux.sh          # Start training
tmux attach -t eg3d_kinface  # Reconnect
```

### `setup_all.sh`
Automated setup for all prerequisites:
```bash
./setup_all.sh  # Runs camera extraction, dataset prep, model download
```

### `generate_multiview.py`
Generate 8-view images after training:
```bash
python generate_multiview.py \
    --network ../../training-runs/.../network-snapshot.pkl \
    --num_views 8 \
    --seeds 0-100
```

---

## 📈 Expected Results

### Training Timeline (RTX 3090)
- 500 kimg: 1-1.5 days (good quality)
- **1000 kimg: 2-3 days (production-ready)** ✅
- 2000 kimg: 4-6 days (best quality)

### Output
- 8 views at angles: -60° to +60°
- 3D-consistent identity preservation
- Ready for kinship verification analysis

---

## 🤝 Team Collaboration

This project is designed for distributed work:
1. **Preprocessing**: Done on local machine (Windows)
2. **Training**: Run on team members' GPU servers (Linux)
3. **Inference**: Generate views for research analysis

All scripts include comprehensive error checking and documentation.

---

## 📝 Citation

If using KinFace-II:
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
  title={Efficient Geometry-aware {3D} Generative Adversarial Networks},
  author={Chan, Eric R and others},
  booktitle={CVPR},
  year={2022}
}
```

---

## 🚀 Ready to Train!

All scripts are production-ready. Your team can start training immediately on GPU servers.

**Questions?** Check the comprehensive [README](project/eg3d_setup/README.md) in `eg3d_setup/`

**License**: Research use only (as per EG3D and KinFace-II licenses)
