"""
Prepare KinFace-II Dataset for EG3D Training
=============================================

This script creates an EG3D-compatible dataset with camera parameters.
Creates both a directory format and an uncompressed ZIP archive.

Output format:
    - dataset.json with image paths and camera labels
    - Uncompressed ZIP for efficient training
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import zipfile
import shutil


class DatasetPreparer:
    """Prepares KinFace-II dataset for EG3D."""
    
    def __init__(self, image_dir, camera_params_path, output_dir):
        """
        Initialize dataset preparer.
        
        Args:
            image_dir: Path to preprocessed images
            camera_params_path: Path to camera parameters JSON
            output_dir: Output directory for EG3D dataset
        """
        self.image_dir = Path(image_dir)
        self.camera_params_path = Path(camera_params_path)
        self.output_dir = Path(output_dir)
        
        # Load camera parameters
        with open(camera_params_path, 'r') as f:
            self.camera_params = json.load(f)
    
    def create_dataset(self, mirror=True):
        """
        Create EG3D dataset with dataset.json.
        
        Args:
            mirror: If True, add horizontally flipped images for augmentation
        """
        print("\n" + "="*60)
        print("Creating EG3D Dataset for KinFace-II")
        print("="*60)
        print(f"Input images: {self.image_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Mirror augmentation: {mirror}")
        print("="*60)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare labels list for dataset.json
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
            image_files = list(relation_dir.glob('*.png')) + \
                         list(relation_dir.glob('*.jpg'))
            
            print(f"\n📁 Processing {relation}: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=relation):
                # Relative path for dataset.json
                rel_path = f"{relation}/{img_path.name}"
                
                # Get camera parameters
                if rel_path not in self.camera_params:
                    print(f"⚠️  Missing camera params for {rel_path}, skipping")
                    continue
                
                camera_label = self.camera_params[rel_path]
                
                # Copy original image
                output_img_path = output_relation_dir / img_path.name
                shutil.copy2(img_path, output_img_path)
                
                # Add to labels
                labels.append([rel_path, camera_label])
                
                # Mirror augmentation
                if mirror:
                    # Flip image horizontally
                    img = Image.open(img_path)
                    img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    # Save mirrored image
                    mirror_name = img_path.stem + "_mirror" + img_path.suffix
                    mirror_path = output_relation_dir / mirror_name
                    img_flipped.save(mirror_path)
                    
                    # Adjust camera parameters for mirrored image
                    # Flip the camera extrinsics (negate x-axis rotation)
                    mirror_camera = camera_label.copy()
                    # Negate first column of rotation matrix (indices 0, 4, 8)
                    mirror_camera[0] = -mirror_camera[0]
                    mirror_camera[4] = -mirror_camera[4]
                    mirror_camera[8] = -mirror_camera[8]
                    
                    # Add mirrored version
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
        """Create uncompressed ZIP archive for efficient training."""
        print("\n" + "="*60)
        print("Creating Uncompressed ZIP Archive")
        print("="*60)
        
        zip_path = self.output_dir.parent / f"{self.output_dir.name}.zip"
        
        print(f"Creating: {zip_path}")
        print("This may take a few minutes...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            # Add all files in output directory
            for file_path in tqdm(list(self.output_dir.rglob('*')), desc="Archiving"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.output_dir)
                    zf.write(file_path, arcname)
        
        # Get size
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        
        print("\n" + "="*60)
        print("✅ ZIP Archive Created")
        print("="*60)
        print(f"Path: {zip_path}")
        print(f"Size: {size_mb:.2f} MB")
        print("="*60)
        
        return zip_path


def main():
    """Main execution."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    image_dir = base_dir / "KinFaceW-II-Processed"
    camera_params_path = base_dir / "project" / "eg3d_setup" / "camera_params" / "camera_params.json"
    output_dir = base_dir / "KinFaceW-II-EG3D"
    
    print("EG3D Dataset Preparation for KinFace-II")
    
    # Check inputs
    if not image_dir.exists():
        print(f"❌ Error: Preprocessed images not found: {image_dir}")
        return
    
    if not camera_params_path.exists():
        print(f"❌ Error: Camera parameters not found: {camera_params_path}")
        print("Please run camera_extraction.py first!")
        return
    
    # Initialize preparer
    preparer = DatasetPreparer(image_dir, camera_params_path, output_dir)
    
    # Create dataset with mirroring
    num_images = preparer.create_dataset(mirror=True)
    
    # Create ZIP archive
    zip_path = preparer.create_zip()
    
    print("\n✅ Dataset preparation complete!")
    print(f"\n📂 Use this for training: {zip_path}")
    print(f"   Total training samples: {num_images}")


if __name__ == "__main__":
    main()
