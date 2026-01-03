"""
KinFace-II Dataset Preprocessing for EG3D
==========================================

This script processes the KinFace-II dataset to make it compatible with EG3D:
1. Face detection & alignment (using MTCNN)
2. Resize to 256×256
3. Save in similar folder structure

Requirements:
    pip install facenet-pytorch pillow numpy opencv-python
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm


class KinFacePreprocessor:
    """Preprocesses KinFace-II dataset for EG3D compatibility."""
    
    def __init__(self, input_dir, output_dir, target_size=256):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Path to KinFaceW-II/images directory
            output_dir: Path to save processed images
            target_size: Target image size (default: 256)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        # Initialize MTCNN for face detection and alignment
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # MTCNN already does alignment (5-point landmark alignment)
        self.mtcnn = MTCNN(
            image_size=target_size,
            margin=20,
            keep_all=False,
            post_process=True,
            device=self.device
        )
        
    def align_face(self, image_path):
        """
        Detect and align face from image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            aligned_face: PIL Image of aligned face (256x256) or None if no face detected
        """
        try:
            # Read image
            img = Image.open(image_path).convert('RGB')
            
            # MTCNN detects, aligns, and crops the face
            # It automatically aligns eyes horizontally using 5 facial landmarks
            face_tensor = self.mtcnn(img)
            
            if face_tensor is None:
                print(f"⚠️ No face detected in {image_path.name}")
                # Fallback: simple resize if no face detected
                return self._simple_resize(img)
            
            # Convert tensor to PIL Image
            # face_tensor is in range [-1, 1], convert to [0, 255]
            face_array = ((face_tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            aligned_face = Image.fromarray(face_array)
            
            return aligned_face
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {str(e)}")
            return None
    
    def _simple_resize(self, img):
        """
        Fallback: Simple center crop and resize if face detection fails.
        
        Args:
            img: PIL Image
            
        Returns:
            resized_img: PIL Image (256x256)
        """
        # Center crop to square
        width, height = img.size
        min_dim = min(width, height)
        
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
        img_cropped = img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        return img_resized
    
    def process_dataset(self):
        """
        Process entire KinFace-II dataset.
        
        Maintains folder structure:
        - father-dau/
        - father-son/
        - mother-dau/
        - mother-son/
        """
        relations = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
        
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'no_face': 0
        }
        
        for relation in relations:
            input_relation_dir = self.input_dir / relation
            output_relation_dir = self.output_dir / relation
            
            if not input_relation_dir.exists():
                print(f"⚠️ Skipping {relation} (not found)")
                continue
            
            # Create output directory
            output_relation_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all image files
            image_files = list(input_relation_dir.glob('*.jpg')) + \
                         list(input_relation_dir.glob('*.png'))
            
            print(f"\n📁 Processing {relation}: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=relation):
                stats['total'] += 1
                
                # Process image
                aligned_face = self.align_face(img_path)
                
                if aligned_face is not None:
                    # Save as PNG (lossless) or JPG
                    output_path = output_relation_dir / f"{img_path.stem}.png"
                    aligned_face.save(output_path, 'PNG', quality=95)
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
        
        # Print summary
        print("\n" + "="*50)
        print("📊 PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total images: {stats['total']}")
        print(f"✅ Successfully processed: {stats['success']}")
        print(f"❌ Failed: {stats['failed']}")
        print(f"Success rate: {stats['success']/stats['total']*100:.2f}%")
        print(f"\n💾 Processed images saved to: {self.output_dir}")
        print("="*50)


def main():
    """Main execution function."""
    # Define paths - go up to project root
    base_dir = Path(__file__).parent.parent.parent  # Up to 3D_Kinship_Verification
    input_dir = base_dir / "KinFaceW-II" / "images"
    output_dir = base_dir / "KinFaceW-II-Processed"
    
    print("="*50)
    print("🚀 KinFace-II Preprocessing for EG3D")
    print("="*50)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: 256×256")
    print("="*50)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        return
    
    # Initialize preprocessor
    preprocessor = KinFacePreprocessor(input_dir, output_dir, target_size=256)
    
    # Process dataset
    preprocessor.process_dataset()
    
    print("\n✅ Done! Dataset is now EG3D-compatible.")
    print(f"📂 Processed images: {output_dir}")


if __name__ == "__main__":
    main()
