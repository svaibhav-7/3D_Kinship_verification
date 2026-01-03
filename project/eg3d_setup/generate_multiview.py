"""
Generate Multi-View Images using EG3D
======================================

This script generates 8-view images from single input faces using a trained
or pretrained EG3D model.

Usage:
    python generate_multiview.py --input image.png --network model.pkl --num_views 8
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import pickle

# Add EG3D to path
SCRIPT_DIR = Path(__file__).parent
EG3D_DIR = SCRIPT_DIR.parent.parent / "eg3d"
sys.path.insert(0, str(EG3D_DIR / "eg3d"))

try:
    import dnnlib
    import legacy
except ImportError:
    print("Error: Could not import EG3D modules")
    print(f"Make sure EG3D is properly set up at: {EG3D_DIR}")
    sys.exit(1)


class MultiViewGenerator:
    """Generates multiple views from a single face image."""
    
    def __init__(self, network_pkl, device='cuda'):
        """
        Initialize the generator.
        
        Args:
            network_pkl: Path to EG3D model pickle file
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load network
        print(f"Loading network from: {network_pkl}")
        with open(network_pkl, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(self.device)
        
        print("Network loaded successfully!")
    
    def generate_camera_params(self, num_views=8, radius=2.7):
        """
        Generate camera parameters for multiple views.
        
        Args:
            num_views: Number of views to generate
            radius: Camera distance from origin
            
        Returns:
            camera_params: List of camera parameter tensors
        """
        camera_params = []
        
        # Generate views at different yaw angles
        angles = np.linspace(-60, 60, num_views)  # degrees
        
        for angle in angles:
            # Convert to radians
            yaw = np.radians(angle)
            pitch = 0  # Keep pitch at 0 (horizontal)
            
            # Camera position on sphere
            cam_pos = np.array([
                radius * np.sin(yaw) * np.cos(pitch),
                radius * np.sin(pitch),
                radius * np.cos(yaw) * np.cos(pitch)
            ])
            
            # Look-at point (origin)
            target = np.array([0, 0, 0])
            up = np.array([0, 1, 0])
            
            # Compute camera-to-world matrix
            forward = target - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(up, forward)
            right = right / np.linalg.norm(right)
            
            up_new = np.cross(forward, right)
            
            # 4x4 extrinsics matrix
            cam2world = np.eye(4, dtype=np.float32)
            cam2world[:3, 0] = right
            cam2world[:3, 1] = up_new
            cam2world[:3, 2] = -forward
            cam2world[:3, 3] = cam_pos
            
            # Intrinsics (normalized)
            focal = 2985.0 / 512
            intrinsics = np.array([
                [focal, 0, 0.5],
                [0, focal, 0.5],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Concatenate
            camera_param = np.concatenate([
                cam2world.flatten(),
                intrinsics.flatten()
            ])
            
            camera_params.append(camera_param)
        
        return camera_params
    
    def generate_from_latent(self, z, camera_params, truncation_psi=0.7):
        """
        Generate images from latent code.
        
        Args:
            z: Latent code tensor
            camera_params: List of camera parameters
            truncation_psi: Truncation value for style mixing
            
        Returns:
            images: List of generated PIL Images
        """
        images = []
        
        with torch.no_grad():
            for cam_param in camera_params:
                # Convert to tensor
                c = torch.from_numpy(cam_param).unsqueeze(0).to(self.device)
                
                # Generate
                img = self.G(z, c, truncation_psi=truncation_psi)['image']
                
                # Convert to PIL
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
                
                images.append(img)
        
        return images
    
    def generate_random_views(self, num_views=8, seed=None):
        """
        Generate random face with multiple views.
        
        Args:
            num_views: Number of views to generate
            seed: Random seed
            
        Returns:
            images: List of PIL Images
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Sample random latent
        z = torch.randn([1, self.G.z_dim]).to(self.device)
        
        # Generate camera parameters
        camera_params = self.generate_camera_params(num_views)
        
        # Generate images
        images = self.generate_from_latent(z, camera_params)
        
        return images


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Generate multi-view images with EG3D')
    parser.add_argument('--network', type=str, required=True,
                       help='Path to EG3D model pickle file')
    parser.add_argument('--num_views', type=int, default=8,
                       help='Number of views to generate')
    parser.add_argument('--seeds', type=str, default='0-3',
                       help='Random seeds (e.g., 0-3 or 0,1,2)')
    parser.add_argument('--outdir', type=str, default='./multiview_output',
                       help='Output directory')
    parser.add_argument('--truncation', type=float, default=0.7,
                       help='Truncation psi')
    
    args = parser.parse_args()
    
    # Parse seeds
    if '-' in args.seeds:
        start, end = map(int, args.seeds.split('-'))
        seeds = range(start, end + 1)
    else:
        seeds = [int(s) for s in args.seeds.split(',')]
    
    print("\n" + "="*60)
    print("EG3D Multi-View Generation")
    print("="*60)
    print(f"Network: {args.network}")
    print(f"Views per face: {args.num_views}")
    print(f"Seeds: {list(seeds)}")
    print(f"Output: {args.outdir}")
    print("="*60 + "\n")
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = MultiViewGenerator(args.network)
    
    # Generate for each seed
    for seed in seeds:
        print(f"\nGenerating views for seed {seed}...")
        
        images = generator.generate_random_views(
            num_views=args.num_views,
            seed=seed
        )
        
        # Save images
        seed_dir = outdir / f"seed{seed:04d}"
        seed_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(images):
            angle = np.linspace(-60, 60, args.num_views)[i]
            img_path = seed_dir / f"view_{i:02d}_angle_{angle:+.1f}.png"
            img.save(img_path)
        
        print(f"  Saved {len(images)} views to: {seed_dir}")
        
        # Create grid
        grid_img = create_image_grid(images, rows=1, cols=args.num_views)
        grid_path = outdir / f"seed{seed:04d}_grid.png"
        grid_img.save(grid_path)
        print(f"  Saved grid to: {grid_path}")
    
    print("\n" + "="*60)
    print("✅ Multi-view generation complete!")
    print(f"📂 Output: {outdir}")
    print("="*60)


def create_image_grid(images, rows, cols):
    """Create a grid of images."""
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    for i, img in enumerate(images):
        grid.paste(img, box=((i % cols) * w, (i // cols) * h))
    
    return grid


if __name__ == "__main__":
    main()
