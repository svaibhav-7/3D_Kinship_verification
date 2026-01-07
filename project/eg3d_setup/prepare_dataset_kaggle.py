"""
EG3D Dataset Preparation for Kaggle
====================================

Kaggle-friendly version without __file__ dependency.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import zipfile
import shutil
import os


class DatasetPreparer:
    """Prepares KinFace-II dataset for EG3D."""
    
    def __init__(self, image_dir, camera_params_path, output_dir):
        """Initialize dataset preparer."""
        self.image_dir = Path(image_dir)
        self.camera_params_path = Path(camera_params_path)
        self.output_dir = Path(output_dir)
        
        # Load camera parameters
        with open(camera_params_path, 'r') as f:
            self.camera_params = json.load(f)
    
    def create_dataset(self, mirror=True):
        """Create EG3D dataset with dataset.json."""
        print("\n" + "="*60)
        print("Creating EG3D Dataset for KinFace-II")
        print("="*60)
        print(f"Input images: {self.image_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Mirror augmentation: {mirror}")
        print("="*60)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare labels list
        labels = []
        relations = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
        
        for relation in relations:
            relation_dir = self.image_dir / relation
            if not relation_dir.exists():
                print(f"\n⚠️  Skipping {relation} (not found)")
                continue
            
            # Create output relation directory
            output_relation_dir = self.output_dir / relation
            output_relation_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all images
            image_files = list(relation_dir.glob('*.png')) + list(relation_dir.glob('*.jpg'))
            
            print(f"\n📁 Processing {relation}: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=relation):
                # Relative path
                rel_path = f"{relation}/{img_path.name}"
                
                if rel_path not in self.camera_params:
                    print(f"⚠️  Missing camera params for {rel_path}")
                    continue
                
                camera_label = self.camera_params[rel_path]
                
                # Copy original image
                output_img_path = output_relation_dir / img_path.name
                shutil.copy2(img_path, output_img_path)
                labels.append([rel_path, camera_label])
                
                # Mirror augmentation
                if mirror:
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
        json_path = self.output_dir / "dataset.json"
        
        with open(json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        print("\n" + "="*60)
        print("📊 Dataset Preparation Summary")
        print("="*60)
        print(f"Total image entries: {len(labels)}")
        print(f"Dataset.json created: {json_path}")
        print("="*60)
        
        return len(labels)
    
    def create_zip(self):
        """Create uncompressed ZIP archive."""
        print("\n" + "="*60)
        print("Creating Uncompressed ZIP Archive")
        print("="*60)
        
        zip_path = self.output_dir.parent / f"{self.output_dir.name}.zip"
        
        print(f"Creating: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for file_path in tqdm(list(self.output_dir.rglob('*')), desc="Archiving"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.output_dir)
                    zf.write(file_path, arcname)
        
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        
        print("\n" + "="*60)
        print("✅ ZIP Archive Created")
        print("="*60)
        print(f"Path: {zip_path}")
        print(f"Size: {size_mb:.2f} MB")
        print("="*60)
        
        return zip_path


# ===== KAGGLE NOTEBOOK USAGE =====

def prepare_dataset_kaggle():
    """Run dataset preparation in Kaggle."""
    image_dir = "KinFaceW-II-Processed"
    camera_params_path = "camera_params/camera_params.json"
    output_dir = "KinFaceW-II-EG3D"
    
    print("EG3D Dataset Preparation for KinFace-II (Kaggle)")
    
    if not os.path.exists(image_dir):
        print(f"❌ Error: Images not found: {image_dir}")
        return
    
    if not os.path.exists(camera_params_path):
        print(f"❌ Error: Camera params not found: {camera_params_path}")
        print("Run camera extraction first!")
        return
    
    # Initialize preparer
    preparer = DatasetPreparer(image_dir, camera_params_path, output_dir)
    
    # Create dataset
    num_images = preparer.create_dataset(mirror=True)
    
    # Create ZIP
    zip_path = preparer.create_zip()
    
    print("\n✅ Dataset preparation complete!")
    print(f"📂 Use this for training: {zip_path}")
    print(f"   Total samples: {num_images}")
    
    return zip_path


if __name__ == "__main__":
    prepare_dataset_kaggle()
