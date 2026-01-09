"""
EG3D Training - Complete Kaggle Pipeline (All-in-One)
======================================================

This script does EVERYTHING from dataset setup to training:
1. Verify dataset
2. Extract camera parameters
3. Prepare EG3D dataset
4. Download pretrained model
5. Verify setup
6. Start training

Usage in Kaggle:
    %run kaggle_train_complete.py
    
Or run as standalone:
    python kaggle_train_complete.py
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
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
}


# ============================================================================
# STEP 1: DATASET VERIFICATION
# ============================================================================

def step1_verify_dataset():
    """Verify and link Kaggle dataset."""
    print("\n" + "="*70)
    print("STEP 1: Dataset Verification")
    print("="*70)
    
    # Check if Kaggle dataset exists
    if os.path.exists(CONFIG["kaggle_dataset"]):
        print(f"✅ Found Kaggle dataset: {CONFIG['kaggle_dataset']}")
        
        # Create symlink if not exists
        if not os.path.exists(CONFIG["local_dataset"]):
            os.symlink(CONFIG["kaggle_dataset"], CONFIG["local_dataset"])
            print(f"✅ Linked to: {CONFIG['local_dataset']}")
        else:
            print(f"✅ Dataset already linked: {CONFIG['local_dataset']}")
    else:
        print(f"❌ ERROR: Kaggle dataset not found!")
        print(f"   Expected: {CONFIG['kaggle_dataset']}")
        print(f"   Make sure dataset is added to notebook!")
        return False
    
    # Count images
    relations = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    total = 0
    for rel in relations:
        rel_path = Path(CONFIG["local_dataset"]) / rel
        if rel_path.exists():
            count = len(list(rel_path.glob('*.png')) + list(rel_path.glob('*.jpg')))
            print(f"   {rel}: {count} images")
            total += count
    
    print(f"\n✅ Total images: {total}")
    return True


# ============================================================================
# STEP 2: CAMERA PARAMETER EXTRACTION
# ============================================================================

def extract_camera_params_for_image(image_path):
    """Extract camera parameters for a single image."""
    # Read image
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


def step2_extract_cameras():
    """Extract camera parameters for all images."""
    print("\n" + "="*70)
    print("STEP 2: Camera Parameter Extraction")
    print("="*70)
    
    input_dir = Path(CONFIG["local_dataset"])
    output_file = Path(CONFIG["camera_params_file"])
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    camera_data = {}
    relations = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    
    total = 0
    for relation in relations:
        relation_dir = input_dir / relation
        if not relation_dir.exists():
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

def step3_prepare_dataset():
    """Prepare EG3D-compatible dataset."""
    print("\n" + "="*70)
    print("STEP 3: EG3D Dataset Preparation")
    print("="*70)
    
    input_dir = Path(CONFIG["local_dataset"])
    output_dir = Path(CONFIG["eg3d_dataset_dir"])
    camera_file = Path(CONFIG["camera_params_file"])
    
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
        
        for img_path in tqdm(image_files, desc=f"{relation} (copy+mirror)"):
            rel_path = f"{relation}/{img_path.name}"
            
            if rel_path not in camera_params:
                continue
            
            camera_label = camera_params[rel_path]
            
            # Copy original
            output_img_path = output_relation_dir / img_path.name
            shutil.copy2(img_path, output_img_path)
            labels.append([rel_path, camera_label])
            
            # Mirror augmentation
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
    
    # Create dataset.json
    dataset_json = {"labels": labels}
    json_path = output_dir / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"\n✅ Total entries: {len(labels)}")
    print(f"✅ Dataset.json: {json_path}")
    
    # Create ZIP
    print("\n📦 Creating ZIP archive...")
    zip_path = Path(CONFIG["eg3d_dataset_zip"])
    
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

def step4_download_pretrained():
    """Download pretrained EG3D model."""
    print("\n" + "="*70)
    print("STEP 4: Download Pretrained Model")
    print("="*70)
    
    model_path = Path(CONFIG["pretrained_model"])
    
    # Check if already exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists: {model_path} ({size_mb:.1f} MB)")
        return True
    
    # Create directory
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    print(f"📥 Downloading from NGC...")
    print(f"   URL: {CONFIG['pretrained_url']}")
    
    try:
        import urllib.request
        
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            if block_num % 100 == 0:
                print(f"   Progress: {percent:.1f}%", end='\r')
        
        urllib.request.urlretrieve(
            CONFIG['pretrained_url'],
            model_path,
            reporthook=download_progress
        )
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Downloaded: {model_path} ({size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n📝 Manual download instructions:")
        print(f"   1. Download from: {CONFIG['pretrained_url']}")
        print(f"   2. Save to: {model_path}")
        return False


# ============================================================================
# STEP 5: VERIFICATION
# ============================================================================

def step5_verify_setup():
    """Verify all setup is complete."""
    print("\n" + "="*70)
    print("STEP 5: Setup Verification")
    print("="*70)
    
    checks = {
        "Dataset ZIP": CONFIG["eg3d_dataset_zip"],
        "Camera params": CONFIG["camera_params_file"],
        "Pretrained model": CONFIG["pretrained_model"],
        "Dataset.json": f"{CONFIG['eg3d_dataset_dir']}/dataset.json",
        "EG3D train.py": "eg3d/eg3d/train.py",
    }
    
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
        print("\n⚠️ Some checks failed. Fix issues before training.")
        return False


# ============================================================================
# STEP 6: START TRAINING
# ============================================================================

def step6_start_training():
    """Start EG3D training."""
    print("\n" + "="*70)
    print("STEP 6: Starting Training")
    print("="*70)
    
    print("\n📋 Training Configuration:")
    print(f"   Dataset: {CONFIG['eg3d_dataset_zip']}")
    print(f"   Pretrained: {CONFIG['pretrained_model']}")
    print(f"   GPUs: {CONFIG['gpus']}")
    print(f"   Batch: {CONFIG['batch']}")
    print(f"   Gamma: {CONFIG['gamma']}")
    print(f"   Training kimg: {CONFIG['kimg']}")
    print(f"   Output: {CONFIG['training_outdir']}")
    
    # Build training command
    cmd = [
        "python", "eg3d/eg3d/train.py",
        f"--outdir={CONFIG['training_outdir']}",
        f"--data={CONFIG['eg3d_dataset_zip']}",
        f"--resume={CONFIG['pretrained_model']}",
        "--cfg=ffhq",
        f"--gpus={CONFIG['gpus']}",
        f"--batch={CONFIG['batch']}",
        f"--gamma={CONFIG['gamma']}",
        "--gen_pose_cond=True",
        "--gpc_reg_prob=0.7",
        "--neural_rendering_resolution_final=128",
        "--aug=ada",
        f"--kimg={CONFIG['kimg']}",
        "--snap=50",
        "--metrics=fid50k_full",
    ]
    
    print("\n🚀 Starting training...")
    print(f"   Command: {' '.join(cmd)}")
    print("\n" + "="*70)
    print("TRAINING OUTPUT")
    print("="*70 + "\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run complete pipeline."""
    print("\n" + "="*70)
    print("EG3D TRAINING - COMPLETE PIPELINE")
    print("="*70)
    print("\nThis script will:")
    print("  1. Verify dataset")
    print("  2. Extract camera parameters")
    print("  3. Prepare EG3D dataset")
    print("  4. Download pretrained model")
    print("  5. Verify setup")
    print("  6. Start training")
    print("\n" + "="*70)
    
    # Run pipeline
    steps = [
        ("Dataset Verification", step1_verify_dataset),
        ("Camera Extraction", step2_extract_cameras),
        ("Dataset Preparation", step3_prepare_dataset),
        ("Download Pretrained", step4_download_pretrained),
        ("Setup Verification", step5_verify_setup),
        ("Start Training", step6_start_training),
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        print(f"\n{'='*70}")
        print(f"Running Step {i}/6: {name}")
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
    print("\nTraining is running...")
    print("Check training-runs/ for outputs")
    print("="*70)
    
    return True


if __name__ == "__main__":
    # Check if running in Kaggle
    if not os.path.exists("/kaggle"):
        print("⚠️ Warning: This script is designed for Kaggle notebooks")
        print("   It may not work correctly in other environments")
    
    # Run pipeline
    success = main()
    
    if success:
        print("\n🎉 Success! Training started.")
    else:
        print("\n❌ Pipeline failed. Check errors above.")
        sys.exit(1)
