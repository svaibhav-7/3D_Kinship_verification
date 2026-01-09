Work in progress!!
aiming for 3d recon based age-invarient kinship## 🚀 Quick Start

### ⭐ NEW: Universal Training Script (Local & Kaggle)

**One script works everywhere!**

```bash
# Local
python train_eg3d_universal.py

# Kaggle
%run train_eg3d_universal.py
```

See [UNIVERSAL_TRAINING.md](UNIVERSAL_TRAINING.md) for details.

---

### For Preprocessing (Already Complete)
```bash
cd project/preprocessing
.\install_deps.bat
.\run_preprocessing.bat
```

### For Training on GPU Server
```bash
cd project/eg3d_setup

# Complete setup
chmod +x setup_all.sh
./setup_all.sh

# Start training with tmux
chmod +x train_tmux.sh
./train_tmux.sh
``` verification

About Kinship
Kinship using facial features refers to identifying or inferring family relationships based on facial resemblance.
Relatives often share similar facial structures (eyes, nose, jawline).
Facial recognition systems can analyze these similarities to predict kinship.
Used in anthropology, genetics, and AI research.
Helpful in missing person identification and family linkage studies
