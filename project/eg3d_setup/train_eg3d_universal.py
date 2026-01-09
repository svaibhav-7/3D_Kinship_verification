"""
EG3D Training - Universal Pipeline (Local & Kaggle)
====================================================

This script works in BOTH environments:
- LOCAL: Windows/Linux with GPU or CPU
- KAGGLE: Kaggle notebooks with free GPU

User selects environment at start:
    1. Local Training
    2. Kaggle Training

Usage:
    python train_eg3d_universal.py
    
Or in Kaggle:
    %run train_eg3d_universal.py
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
import platform
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    print("⚠️ OpenCV not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2


# ============================================================================
# ENVIRONMENT DETECTION & CONFIGURATION
# ============================================================================

class EnvironmentConfig:
    """Detects and configures environment (Local or Kaggle)."""
    
    def __init__(self):
        self.is_kaggle = os.path.exists("/kaggle")
        self.is_windows = platform.system() == "Windows"
        self.is_linux = platform.system() == "Linux"
        
        # Detect environment automatically
        if self.is_kaggle:
            self.env_type = "kaggle"
            print("🔍 Detected: Kaggle Notebook")
        else:
            self.env_type = "local"
            print(f"🔍 Detected: Local Machine ({platform.system()})")
    
    def select_environment(self):
        """Let user select environment."""
        print("\n" + "="*70)
        print("ENVIRONMENT SELECTION")
        print("="*70)
        print("Select training environment:")
        print("  1. Local Training (your machine)")
        print("  2. Kaggle Training (Kaggle notebook)")
        print("="*70)
        
        # Auto-select if in Kaggle
        if self.is_kaggle:
            print("🔔 Running in Kaggle - Using Kaggle configuration")
            choice = "2"
        else:
            choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            self.env_type = "local"
            print("✅ Selected: LOCAL Training")
        elif choice == "2":
            self.env_type = "kaggle"
            print("✅ Selected: KAGGLE Training")
        else:
            print("❌ Invalid choice. Defaulting to LOCAL.")
            self.env_type = "local"
        
        return self.env_type
    
    def get_config(self):
        """Get environment-specific configuration."""
        if self.env_type == "kaggle":
            return self._kaggle_config()
        else:
            return self._local_config()
    
    def _kaggle_config(self):
        """Kaggle-specific configuration."""
        return {
            # Dataset paths
            "kaggle_dataset": "/kaggle/input/kinface-ii-processed-256/KinFaceW-II-Processed",
            "local_dataset": "KinFaceW-II-Processed",
            
            # Output paths
            "camera_params_dir": "camera_params",
            "camera_params_file": "camera_params/camera_params.json",
            "eg3d_dataset_dir": "KinFaceW-II-EG3D",
            "eg3d_dataset_zip": "KinFaceW-II-EG3D.zip",
            
            # Pretrained model
            "pretrained_dir": "eg3d/pretrained",
            "pretrained_model": "eg3d/pretrained/ffhqrebalanced512-128.pkl",
            "pretrained_url": "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-128.pkl",
            
            # Training parameters
            "training_outdir": "training-runs",
            "gpus": 1,
            "batch": 4,
            "gamma": 5,
            "kimg": 1000,
            
            # Environment
            "use_symlink": True,
            "env_type": "kaggle",
        }
    
    def _local_config(self):
        """Local machine configuration."""
        # Detect base directory
        base_dir = Path.cwd()
        
        # Check if we're in project root
        if not (base_dir / "KinFaceW-II-Processed").exists():
            # Try to find it
            possible_paths = [
                base_dir / "KinFaceW-II-Processed",
                base_dir.parent / "KinFaceW-II-Processed",
                base_dir / ".." / "KinFaceW-II-Processed",
            ]
            
            for p in possible_paths:
                if p.exists():
                    base_dir = p.parent
                    break
        
        return {
            # Dataset paths (local)
            "kaggle_dataset": None,
            "local_dataset": str(base_dir / "KinFaceW-II-Processed"),
            
            # Output paths
            "camera_params_dir": str(base_dir / "project" / "eg3d_setup" / "camera_params"),
            "camera_params_file": str(base_dir / "project" / "eg3d_setup" / "camera_params" / "camera_params.json"),
            "eg3d_dataset_dir": str(base_dir / "KinFaceW-II-EG3D"),
            "eg3d_dataset_zip": str(base_dir / "KinFaceW-II-EG3D.zip"),
            
            # Pretrained model
            "pretrained_dir": str(base_dir / "eg3d" / "pretrained"),
            "pretrained_model": str(base_dir / "eg3d" / "pretrained" / "ffhqrebalanced512-128.pkl"),
            "pretrained_url": "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-128.pkl",
            
            # Training parameters (adjustable for local GPU)
            "training_outdir": str(base_dir / "training-runs"),
            "gpus": 1,  # Auto-detect GPUs
            "batch": 4,  # Adjust based on GPU memory
            "gamma": 5,
            "kimg": 1000,
            
            # Environment
            "use_symlink": False,
            "env_type": "local",
        }


# ============================================================================
# STEP 1: DATASET VERIFICATION
# ============================================================================

def step1_verify_dataset(config):
    """Verify and link dataset."""
    print("\n" + "="*70)
    print("STEP 1: Dataset Verification")
    print("="*70)
    
    # Kaggle: Check Kaggle dataset and create symlink
    if config["env_type"] == "kaggle":
        if os.path.exists(config["kaggle_dataset"]):
            print(f"✅ Found Kaggle dataset: {config['kaggle_dataset']}")
            
            if not os.path.exists(config["local_dataset"]):
                os.symlink(config["kaggle_dataset"], config["local_dataset"])
                print(f"✅ Linked to: {config['local_dataset']}")
            else:
                print(f"✅ Dataset already linked")
        else:
            print(f"❌ ERROR: Kaggle dataset not found!")
            print(f"   Expected: {config['kaggle_dataset']}")
            return False
    
    # Local: Check if dataset exists
    else:
        if os.path.exists(config["local_dataset"]):
            print(f"✅ Found local dataset: {config['local_dataset']}")
        else:
            print(f"❌ ERROR: Dataset not found!")
            print(f"   Expected: {config['local_dataset']}")
            print(f"\n   Make sure KinFaceW-II-Processed folder exists")
            return False
    
    # Count images
    relations = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    total = 0
    
    for rel in relations:
        rel_path = Path(config["local_dataset"]) / rel
        if rel_path.exists():
            count = len(list(rel_path.glob('*.png')) + list(rel_path.glob('*.jpg')))
            print(f"   {rel}: {count} images")
            total += count
    
    if total == 0:
        print("❌ ERROR: No images found!")
        return False
    
    print(f"\n✅ Total images: {total}")
    return True


# ============================================================================
# STEP 2: CAMERA PARAMETER EXTRACTION
# ============================================================================

def extract_camera_params_for_image(image_path):
    """Extract camera parameters for a single image."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Default front-facing camera
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[2, 3] = 2.7
        
        focal = 2985.0 / 512
        intrinsics = np.array([
            [focal, 0.0, 0.5],
            [0.0, focal, 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return np.concatenate([extrinsics.flatten(), intrinsics.flatten()])
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return None


def step2_extract_cameras(config):
    """Extract camera parameters for all images."""
    print("\n" + "="*70)
    print("STEP 2: Camera Parameter Extraction")
    print("="*70)
    
    input_dir = Path(config["local_dataset"])
    output_file = Path(config["camera_params_file"])
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    camera_data = {}
    relations = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    
    total = 0
    for relation in relations:
        relation_dir = input_dir / relation
        if not relation_dir.exists():
            print(f"⚠️ Skipping {relation} (not found)")
            continue
        
        image_files = list(relation_dir.glob('*.png')) + list(relation_dir.glob('*.jpg'))
        print(f"\n📁 {relation}: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=relation):
            rel_path = f"{relation}/{img_path.name}"
            camera_params = extract_camera_params_for_image(img_path)
            
            if camera_params is not None:
                camera_data[rel_path] = camera_params.tolist()
                total += 1
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(camera_data, f, indent=2)
    
    print(f"\n✅ Processed: {total} images")
    print(f"✅ Saved to: {output_file}")
    return True


# ============================================================================
# STEP 3: EG3D DATASET PREPARATION
# ============================================================================

def step3_prepare_dataset(config):
    """Prepare EG3D-compatible dataset."""
    print("\n" + "="*70)
    print("STEP 3: EG3D Dataset Preparation")
    print("="*70)
    
    input_dir = Path(config["local_dataset"])
    output_dir = Path(config["eg3d_dataset_dir"])
    camera_file = Path(config["camera_params_file"])
    
    # Load camera params
    with open(camera_file, 'r') as f:
        camera_params = json.load(f)
    
    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = []
    relations = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    
    for relation in relations:
        relation_dir = input_dir / relation
        if not relation_dir.exists():
            continue
        
        output_relation_dir = output_dir / relation
        output_relation_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(relation_dir.glob('*.png')) + list(relation_dir.glob('*.jpg'))
        print(f"\n📁 {relation}: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"{relation}"):
            rel_path = f"{relation}/{img_path.name}"
            
            if rel_path not in camera_params:
                continue
            
            camera_label = camera_params[rel_path]
            
            # Copy original
            output_img_path = output_relation_dir / img_path.name
            shutil.copy2(img_path, output_img_path)
            labels.append([rel_path, camera_label])
            
            # Mirror augmentation
            try:
                img = Image.open(img_path)
                img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                mirror_name = img_path.stem + "_mirror" + img_path.suffix
                mirror_path = output_relation_dir / mirror_name
                img_flipped.save(mirror_path)
                
                # Adjust camera for mirror
                mirror_camera = camera_label.copy()
                mirror_camera[0] = -mirror_camera[0]
                mirror_camera[4] = -mirror_camera[4]
                mirror_camera[8] = -mirror_camera[8]
                
                mirror_rel_path = f"{relation}/{mirror_name}"
                labels.append([mirror_rel_path, mirror_camera])
            except Exception as e:
                print(f"⚠️ Mirror failed for {img_path.name}: {e}")
    
    # Create dataset.json
    dataset_json = {"labels": labels}
    json_path = output_dir / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"\n✅ Total entries: {len(labels)}")
    print(f"✅ Dataset.json: {json_path}")
    
    # Create ZIP
    print("\n📦 Creating ZIP archive...")
    zip_path = Path(config["eg3d_dataset_zip"])
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for file_path in tqdm(list(output_dir.rglob('*')), desc="Archiving"):
            if file_path.is_file():
                arcname = file_path.relative_to(output_dir)
                zf.write(file_path, arcname)
    
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"✅ ZIP created: {zip_path} ({size_mb:.1f} MB)")
    
    return True


# ============================================================================
# STEP 4: DOWNLOAD PRETRAINED MODEL
# ============================================================================

def step4_download_pretrained(config):
    """Download pretrained EG3D model."""
    print("\n" + "="*70)
    print("STEP 4: Download Pretrained Model")
    print("="*70)
    
    model_path = Path(config["pretrained_model"])
    
    # Check if already exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        if size_mb > 500:  # Reasonable size check
            print(f"✅ Model already exists: {model_path} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"⚠️ Model file too small ({size_mb:.1f} MB), re-downloading...")
    
    # Create directory
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    print(f"📥 Downloading from NGC...")
    print(f"   This may take 5-10 minutes (~650 MB)")
    
    try:
        import urllib.request
        
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            if block_num % 50 == 0:
                print(f"   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
        
        urllib.request.urlretrieve(
            config['pretrained_url'],
            model_path,
            reporthook=download_progress
        )
        
        print()  # New line after progress
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Downloaded: {model_path} ({size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n📝 Manual download instructions:")
        print(f"   1. Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d")
        print(f"   2. Download: ffhqrebalanced512-128.pkl")
        print(f"   3. Place at: {model_path}")
        return False


# ============================================================================
# STEP 5: VERIFICATION
# ============================================================================

def step5_verify_setup(config):
    """Verify all setup is complete."""
    print("\n" + "="*70)
    print("STEP 5: Setup Verification")
    print("="*70)
    
    checks = {
        "Dataset ZIP": config["eg3d_dataset_zip"],
        "Camera params": config["camera_params_file"],
        "Pretrained model": config["pretrained_model"],
        "Dataset.json": f"{config['eg3d_dataset_dir']}/dataset.json",
    }
    
    # Add EG3D train.py check
    eg3d_train_paths = [
        "eg3d/eg3d/train.py",
        "../eg3d/eg3d/train.py",
        Path(config["eg3d_dataset_zip"]).parent / "eg3d" / "eg3d" / "train.py",
    ]
    
    train_py_found = False
    for p in eg3d_train_paths:
        if os.path.exists(p):
            checks["EG3D train.py"] = str(p)
            train_py_found = True
            break
    
    if not train_py_found:
        checks["EG3D train.py"] = "eg3d/eg3d/train.py (NOT FOUND)"
    
    all_good = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_good = False
    
    print("="*70)
    
    if all_good:
        print("\n🎉 All checks passed! Ready to train!")
        return True
    else:
        print("\n⚠️ Some checks failed.")
        if not train_py_found:
            print("\n💡 EG3D not found. Clone it:")
            print("   git clone https://github.com/NVlabs/eg3d.git")
        return False


# ============================================================================
# STEP 6: START TRAINING
# ============================================================================

def step6_start_training(config):
    """Start EG3D training."""
    print("\n" + "="*70)
    print("STEP 6: Starting Training")
    print("="*70)
    
    print("\n📋 Training Configuration:")
    print(f"   Environment: {config['env_type'].upper()}")
    print(f"   Dataset: {config['eg3d_dataset_zip']}")
    print(f"   Pretrained: {config['pretrained_model']}")  
    print(f"   GPUs: {config['gpus']}")
    print(f"   Batch: {config['batch']}")
    print(f"   Gamma: {config['gamma']}")
    print(f"   Training kimg: {config['kimg']}")
    print(f"   Output: {config['training_outdir']}")
    
    # Find train.py
    train_script = "eg3d/eg3d/train.py"
    if not os.path.exists(train_script):
        print(f"\n❌ ERROR: {train_script} not found!")
        print("   Clone EG3D first: git clone https://github.com/NVlabs/eg3d.git")
        return False
    
    # Build training command
    cmd = [
        sys.executable, train_script,
        f"--outdir={config['training_outdir']}",
        f"--data={config['eg3d_dataset_zip']}",
        f"--resume={config['pretrained_model']}",
        "--cfg=ffhq",
        f"--gpus={config['gpus']}",
        f"--batch={config['batch']}",
        f"--gamma={config['gamma']}",
        "--gen_pose_cond=True",
        "--gpc_reg_prob=0.7",
        "--neural_rendering_resolution_final=128",
        "--aug=ada",
        f"--kimg={config['kimg']}",
        "--snap=50",
        "--metrics=fid50k_full",
    ]
    
    print("\n🚀 Starting training...")
    print(f"\nCommand: {' '.join(cmd)}")
    print("\n" + "="*70)
    print("TRAINING OUTPUT")
    print("="*70 + "\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run complete pipeline."""
    print("\n" + "="*70)
    print("EG3D TRAINING - UNIVERSAL PIPELINE")
    print("Supports: Local & Kaggle Environments")
    print("="*70)
    
    # Initialize environment
    env = EnvironmentConfig()
    env.select_environment()
    config = env.get_config()
    
    print("\n📋 Configuration:")
    print(f"   Environment: {config['env_type'].upper()}")
    print(f"   Platform: {platform.system()}")
    print(f"   Python: {sys.version.split()[0]}")
    
    # Pipeline steps
    steps = [
        ("Dataset Verification", lambda: step1_verify_dataset(config)),
        ("Camera Extraction", lambda: step2_extract_cameras(config)),
        ("Dataset Preparation", lambda: step3_prepare_dataset(config)),
        ("Download Pretrained", lambda: step4_download_pretrained(config)),
        ("Setup Verification", lambda: step5_verify_setup(config)),
        ("Start Training", lambda: step6_start_training(config)),
    ]
    
    # Run pipeline
    for i, (name, func) in enumerate(steps, 1):
        print(f"\n{'='*70}")
        print(f"Step {i}/6: {name}")
        print(f"{'='*70}")
        
        try:
            success = func()
            if not success:
                print(f"\n❌ Step {i} failed: {name}")
                print("Fix the issue and run again.")
                return False
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        
        if success:
            print("\n🎉 Success! Training started.")
            sys.exit(0)
        else:
            print("\n❌ Pipeline failed. Check errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
