#!/usr/bin/env python3
"""
Model Downloader for DeepFake Detection App
Downloads large model files that can't be stored in the repository
"""

import os
import requests
import hashlib
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model download configurations
MODEL_DOWNLOADS = {
    'tf_model.h5': {
        'url': 'https://github.com/ondyari/FaceForensics/releases/download/v1.0/tf_model.h5',
        'size': 97990000,  # Approximate size in bytes
        'sha256': None,  # Add checksum if available
        'description': 'ResNet50-based deepfake detection model'
    },
    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5': {
        'url': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'size': 79800000,  # Approximate size in bytes
        'sha256': None,  # Add checksum if available
        'description': 'Xception base weights (ImageNet, no top)'
    }
}

def download_file(url, filepath, expected_size=None, chunk_size=8192):
    """Download a file with progress bar"""
    try:
        logger.info(f"Downloading {filepath.name} from {url}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', expected_size or 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=f"Downloading {filepath.name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"✓ Successfully downloaded {filepath.name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {filepath.name}: {str(e)}")
        if filepath.exists():
            filepath.unlink()  # Remove partial download
        return False

def verify_file(filepath, expected_size=None, expected_sha256=None):
    """Verify downloaded file integrity"""
    if not filepath.exists():
        return False
    
    # Check file size
    if expected_size:
        actual_size = filepath.stat().st_size
        if abs(actual_size - expected_size) > expected_size * 0.1:  # 10% tolerance
            logger.warning(f"Size mismatch for {filepath.name}: expected ~{expected_size}, got {actual_size}")
            return False
    
    # Check SHA256 if provided
    if expected_sha256:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_sha256 = sha256_hash.hexdigest()
        if actual_sha256 != expected_sha256:
            logger.error(f"SHA256 mismatch for {filepath.name}")
            return False
    
    return True

def download_models(force_download=False):
    """Download all required model files"""
    logger.info("=== Model Download Check ===")
    
    success_count = 0
    total_models = len(MODEL_DOWNLOADS)
    
    for filename, config in MODEL_DOWNLOADS.items():
        filepath = Path(filename)
        
        # Check if file already exists and is valid
        if filepath.exists() and not force_download:
            if verify_file(filepath, config.get('size'), config.get('sha256')):
                logger.info(f"✓ {filename} already exists and is valid")
                success_count += 1
                continue
            else:
                logger.warning(f"⚠ {filename} exists but is invalid, re-downloading...")
                filepath.unlink()
        
        # Download the file
        logger.info(f"Downloading {config['description']}")
        if download_file(config['url'], filepath, config.get('size')):
            if verify_file(filepath, config.get('size'), config.get('sha256')):
                success_count += 1
            else:
                logger.error(f"✗ Downloaded {filename} failed verification")
        else:
            logger.error(f"✗ Failed to download {filename}")
    
    logger.info(f"=== Download Complete: {success_count}/{total_models} models ready ===")
    return success_count == total_models

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download model files for DeepFake Detection App')
    parser.add_argument('--force', action='store_true', help='Force re-download even if files exist')
    args = parser.parse_args()
    
    success = download_models(force_download=args.force)
    if success:
        logger.info("🎉 All models downloaded successfully!")
        exit(0)
    else:
        logger.error("❌ Some models failed to download")
        exit(1)

if __name__ == '__main__':
    main()