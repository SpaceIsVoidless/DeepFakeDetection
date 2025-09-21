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
# Note: We're using working URLs and the app gracefully handles missing models
MODEL_DOWNLOADS = {
    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5': {
        'url': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'size': 79800000,  # Approximate size in bytes
        'sha256': None,  # Add checksum if available
        'description': 'Xception base weights (ImageNet, no top)'
    }
    # tf_model.h5 is not publicly available, but ResNet50 works with ImageNet weights
    # The app will function perfectly with just the Xception weights and built-in MesoNet models
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
    """Download all available model files"""
    logger.info("=== Model Download Check ===")
    
    if not MODEL_DOWNLOADS:
        logger.info("ℹ️  No models configured for download")
        logger.info("✅ App will use built-in models (MesoNet) and ImageNet base weights")
        return True
    
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
                logger.info(f"✅ Successfully downloaded and verified {filename}")
            else:
                logger.error(f"✗ Downloaded {filename} failed verification")
        else:
            logger.warning(f"⚠️  Failed to download {filename} - app will continue without it")
    
    logger.info(f"=== Download Complete: {success_count}/{total_models} optional models downloaded ===")
    
    # Always return True since the app can work without these optional downloads
    if success_count > 0:
        logger.info("🎉 Some models downloaded successfully!")
    else:
        logger.info("ℹ️  No additional models downloaded, using built-in models")
    
    logger.info("✅ App is ready to start with available models")
    return True

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download model files for DeepFake Detection App')
    parser.add_argument('--force', action='store_true', help='Force re-download even if files exist')
    args = parser.parse_args()
    
    logger.info("🚀 DeepFake Detection App - Model Downloader")
    logger.info("=" * 50)
    
    success = download_models(force_download=args.force)
    
    # Check what models are available
    logger.info("\n📊 Model Status Summary:")
    
    # Check built-in models
    builtin_models = ['Meso4_DF.h5', 'Meso4_F2F.h5', 'MesoInception_DF.h5', 'MesoInception_F2F.h5']
    for model in builtin_models:
        if Path(model).exists():
            size = Path(model).stat().st_size / 1024 / 1024
            logger.info(f"✅ {model} ({size:.2f}MB) - Built-in")
        else:
            logger.info(f"❌ {model} - Missing")
    
    # Check downloaded models
    for filename in MODEL_DOWNLOADS.keys():
        if Path(filename).exists():
            size = Path(filename).stat().st_size / 1024 / 1024
            logger.info(f"✅ {filename} ({size:.2f}MB) - Downloaded")
        else:
            logger.info(f"⚠️  {filename} - Not available (app will use base weights)")
    
    logger.info("\n🎯 Ready to start the DeepFake Detection App!")
    logger.info("   Run: python app.py")
    exit(0)

if __name__ == '__main__':
    main()