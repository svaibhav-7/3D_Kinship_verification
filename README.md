# 3D Kinship Verification using EG3D

## 📝 Overview
This project aims to perform kinship verification (identifying family relationships like father-son, mother-daughter) by leveraging 3D facial reconstruction using **EG3D** (Efficient Geometry-aware 3D Generative Adversarial Networks). By generating 3D-consistent face images from the **KinFaceW-II** dataset, we aim to achieve age-invariant kinship verification.

## ✨ Features
- **Universal Training Script**: A single script (`train_eg3d_universal.py`) that works on both Local machines and Kaggle notebooks.
- **Automated Preprocessing**: Scripts to detect faces, align them, and resize them for EG3D.
- **3D Multi-View Generation**: Generate 3D-consistent views of faces.
- **Support for KinFaceW-II Dataset**: Specifically tailored for this dataset.

## 📂 Project Structure
```
.
├── KinFaceW-II-Processed/       # Processed dataset (generated output)
├── project/
│   ├── eg3d_setup/              # Training scripts and EG3D setup
│   │   ├── train_eg3d_universal.py  # Main universal training script
│   │   ├── train_tmux.sh            # Training with tmux (Linux)
│   │   └── ...
│   ├── preprocessing/           # Preprocessing scripts
│   │   ├── preprocess_kinface.py    # Main preprocessing script
│   │   └── ...
│   └── ...
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.8+**
- **NVIDIA GPU** (for training) with CUDA support.
- **Git**

### 2. Preprocessing
Before training, you need to preprocess the KinFaceW-II dataset to align and resize images for EG3D.

**Steps:**
1. Navigate to the preprocessing directory:
   ```bash
   cd project/preprocessing
   ```
2. Install dependencies:
   ```bash
   # Windows
   ./install_deps.bat

   # Linux/Mac
   pip install -r ../requirements.txt
   ```
3. Run preprocessing:
   ```bash
   # Windows
   ./run_preprocessing.bat

   # Linux/Mac
   python preprocess_kinface.py
   ```

For more details, see the [Preprocessing Guide](project/preprocessing/README.md).

### 3. Training (EG3D)

We provide a **Universal Training Script** that handles dataset preparation, model downloading, and training automatically. It detects whether you are running locally or on Kaggle.

#### Option A: Local Training
1. Navigate to the setup directory:
   ```bash
   cd project/eg3d_setup
   ```
2. Run the universal script:
   ```bash
   python train_eg3d_universal.py
   ```
3. Follow the on-screen prompts to select "Local Training".

#### Option B: Kaggle Training
Upload `project/eg3d_setup/train_eg3d_universal.py` to your Kaggle notebook or download it directly:

```python
!wget https://raw.githubusercontent.com/svaibhav-7/3D_Kinship_verification/main/project/eg3d_setup/train_eg3d_universal.py
%run train_eg3d_universal.py
```

For more details on training, see:
- [Universal Training Guide](project/eg3d_setup/UNIVERSAL_TRAINING.md)
- [EG3D Setup Guide](project/eg3d_setup/README.md)

## 📄 Documentation
- [Preprocessing Details](project/preprocessing/README.md): Detailed explanation of face alignment and resizing.
- [EG3D Training Setup](project/eg3d_setup/README.md): Comprehensive guide on training configuration, tmux usage, and troubleshooting.
- [Universal Training Script](project/eg3d_setup/UNIVERSAL_TRAINING.md): Explains how the single script works across environments.

## 📚 References
- **EG3D**: [Efficient Geometry-aware 3D Generative Adversarial Networks](https://nvlabs.github.io/eg3d/)
- **KinFaceW-II Dataset**: [Kinship Face Verification](https://www.kinface.org/)

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
