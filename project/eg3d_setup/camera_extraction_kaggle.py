"""
Camera Parameter Extraction for KinFace-II Dataset (Kaggle Version)
=====================================================================

This is a Kaggle notebook-friendly version that doesn't use __file__.
Run this in Kaggle notebook cells.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

class CameraExtractor:
    """Extracts camera parameters from face images."""
    
    def __init__(self):
        """Initialize the camera extractor."""
        print("Using default front-facing camera parameters")
        
    def extract_camera_params(self, image_path):
        """
        Extract camera parameters from a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            camera_params: 25-length numpy array [16 extrinsics + 9 intrinsics]
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        # Default front-facing camera parameters
        # Extrinsics: 4x4 camera-to-world matrix (identity with z offset)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[2, 3] = 2.7  # Camera distance
        
        # Intrinsics: 3x3 normalized intrinsics
        focal = 2985.0 / 512  # Normalized focal length (~5.8)
        intrinsics = np.array([
            [focal, 0.0, 0.5],
            [0.0, focal, 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Flatten and concatenate
        camera_params = np.concatenate([
            extrinsics.flatten(),
            intrinsics.flatten()
        ])
        
        return camera_params
    
    def process_dataset(self, input_dir, output_path):
        """
        Process entire dataset and save camera parameters.
        
        Args:
            input_dir: Path to preprocessed KinFace images
            output_path: Path to save camera parameters JSON
        """
        input_dir = Path(input_dir)
        output_path = Path(output_path)
        
        camera_data = {}
        relations = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
        
        print("\n" + "="*60)
        print("Extracting Camera Parameters for KinFace-II")
        print("="*60)
        
        total_images = 0
        for relation in relations:
            relation_dir = input_dir / relation
            if not relation_dir.exists():
                print(f"⚠️  Skipping {relation} (not found)")
                continue
            
            image_files = list(relation_dir.glob('*.png')) + \
                         list(relation_dir.glob('*.jpg'))
            
            print(f"\n📁 Processing {relation}: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=relation):
                # Extract relative path
                rel_path = f"{relation}/{img_path.name}"
                
                # Extract camera parameters
                camera_params = self.extract_camera_params(img_path)
                
                if camera_params is not None:
                    camera_data[rel_path] = camera_params.tolist()
                    total_images += 1
        
        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(camera_data, f, indent=2)
        
        print("\n" + "="*60)
        print("📊 Camera Extraction Summary")
        print("="*60)
        print(f"Total images processed: {total_images}")
        print(f"Camera parameters saved to: {output_path}")
        print("="*60)
        print("\n⚠️  NOTE: Using default camera parameters")


# ===== KAGGLE NOTEBOOK USAGE =====
# Run this in your Kaggle notebook cell:

def extract_cameras_kaggle():
    """Run camera extraction in Kaggle notebook."""
    # Define paths (adjust if needed)
    input_dir = "KinFaceW-II-Processed"  # or "/kaggle/input/kinface-ii-processed-256/KinFaceW-II-Processed"
    output_dir = "camera_params"
    output_path = f"{output_dir}/camera_params.json"
    
    print("Camera Parameter Extraction (Kaggle)")
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Error: Preprocessed dataset not found: {input_dir}")
        print("Make sure dataset is linked!")
        return
    
    # Initialize extractor
    extractor = CameraExtractor()
    
    # Process dataset
    extractor.process_dataset(input_dir, output_path)
    
    print("\n✅ Done!")
    return output_path


# Call this function in your Kaggle notebook
if __name__ == "__main__":
    extract_cameras_kaggle()
