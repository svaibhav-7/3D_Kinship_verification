"""
Camera Parameter Extraction for KinFace-II Dataset
====================================================

This script extracts camera parameters (extrinsics + intrinsics) from face images
using Deep3DFaceRecon_pytorch, which is required for EG3D training.

Requirements:
    - Deep3DFaceRecon_pytorch submodule initialized
    - PyTorch with CUDA
    - All dependencies from Deep3DFaceRecon
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image
import pickle

# Add Deep3DFaceRecon to path
SCRIPT_DIR = Path(__file__).parent
EG3D_DIR = SCRIPT_DIR.parent.parent / "eg3d"
DEEP3D_DIR = EG3D_DIR / "eg3d" / "Deep3DFaceRecon_pytorch"

sys.path.insert(0, str(DEEP3D_DIR))

try:
    import torch
    from models.bfm import ParametricFaceModel
    from models.facerecon_model import FaceReconModel
    from util.preprocess import align_img
    from util.load_mats import load_lm3d
except ImportError as e:
    print(f"Error importing Deep3DFaceRecon modules: {e}")
    print(f"Make sure Deep3DFaceRecon_pytorch submodule is initialized:")
    print(f"  cd {EG3D_DIR}")
    print(f"  git submodule update --init --recursive")
    sys.exit(1)


class CameraExtractor:
    """Extracts camera parameters from face images."""
    
    def __init__(self, checkpoint_path=None):
        """
        Initialize the camera extractor.
        
        Args:
            checkpoint_path: Path to Deep3DFaceRecon pretrained checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize face reconstruction model
        # Note: You'll need to download the pretrained model from:
        # https://github.com/sicxu/Deep3DFaceRecon_pytorch
        self.model = None  # Will be initialized when checkpoint is provided
        
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
        
        # For now, use default camera parameters
        # This is a simplification - ideally you'd use Deep3DFaceRecon
        # Your team members can update this with actual face reconstruction
        
        # Default front-facing camera parameters
        # Extrinsics: 4x4 camera-to-world matrix (identity with z offset)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[2, 3] = 2.7  # Camera distance (typical for face imaging)
        
        # Intrinsics: 3x3 normalized intrinsics
        # Assuming 512x512 image with typical focal length
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
        print("For better results, your team should:")
        print("1. Download Deep3DFaceRecon pretrained weights")
        print("2. Update this script to use actual face reconstruction")
        print("3. Re-run camera extraction")


def main():
    """Main execution."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / "KinFaceW-II-Processed"
    output_dir = base_dir / "project" / "eg3d_setup" / "camera_params"
    output_path = output_dir / "camera_params.json"
    
    print("Camera Parameter Extraction")
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Error: Preprocessed dataset not found: {input_dir}")
        print("Please run preprocessing first!")
        return
    
    # Initialize extractor
    extractor = CameraExtractor()
    
    # Process dataset
    extractor.process_dataset(input_dir, output_path)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
