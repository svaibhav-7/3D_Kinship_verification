"""
Download Pretrained EG3D Model
===============================

Downloads the pretrained FFHQ model for fine-tuning on KinFace-II.

Models available:
    - ffhqrebalanced512-128.pkl (recommended)
    - ffhq512-128.pkl
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                  reporthook=t.update_to)


def main():
    """Main download function."""
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / "eg3d" / "pretrained"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("EG3D Pretrained Model Download")
    print("="*60)
    
    # Model URLs from NGC Catalog
    # Note: These URLs may change. Check:
    # https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d
    models = {
        'ffhqrebalanced512-128.pkl': {
            'url': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-128.pkl',
            'description': 'FFHQ Rebalanced 512x512, 128 neural rendering (RECOMMENDED)',
            'size': '~650 MB'
        },
        'ffhq512-128.pkl': {
            'url': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhq512-128.pkl',
            'description': 'FFHQ 512x512, 128 neural rendering',
            'size': '~650 MB'
        }
    }
    
    print("\nAvailable models:")
    for i, (name, info) in enumerate(models.items(), 1):
        print(f"{i}. {name}")
        print(f"   {info['description']}")
        print(f"   Size: {info['size']}\n")
    
    # Download recommended model
    model_name = 'ffhqrebalanced512-128.pkl'
    model_info = models[model_name]
    output_path = models_dir / model_name
    
    if output_path.exists():
        print(f"✅ Model already exists: {output_path}")
        print("Skipping download.")
    else:
        print(f"📥 Downloading: {model_name}")
        print(f"URL: {model_info['url']}")
        print(f"Destination: {output_path}")
        print("")
        
        try:
            download_url(model_info['url'], output_path)
            print(f"\n✅ Download complete: {output_path}")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print("\nManual download instructions:")
            print(f"1. Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d")
            print(f"2. Download: {model_name}")
            print(f"3. Place in: {models_dir}/")
            return
    
    # Verify file size
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n📊 Model size: {size_mb:.2f} MB")
        
        if size_mb < 500:
            print("⚠️  Warning: File size seems small. Download may be incomplete.")
        else:
            print("✅ File size looks good!")
    
    print("\n" + "="*60)
    print("Ready for training!")
    print("="*60)


if __name__ == "__main__":
    main()
