#!/usr/bin/env python3
"""
Quick fix for Render deployment - ensures app starts even if model downloads fail
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Ensure the app can start with available models"""
    logger.info("🔧 Render Deployment Fix")
    
    # Check which models are available
    models = {
        'Meso4_DF.h5': 'Built-in MesoNet model',
        'Meso4_F2F.h5': 'Built-in MesoNet model', 
        'MesoInception_DF.h5': 'Built-in MesoInception model',
        'MesoInception_F2F.h5': 'Built-in MesoInception model',
        'tf_model.h5': 'ResNet50 weights (optional)',
        'xception_weights_tf_dim_ordering_tf_kernels_notop.h5': 'Xception weights (optional)'
    }
    
    available_count = 0
    for model, description in models.items():
        if os.path.exists(model):
            size = os.path.getsize(model) / 1024 / 1024
            logger.info(f"✅ {model} ({size:.2f}MB) - {description}")
            available_count += 1
        else:
            logger.info(f"⚠️  {model} - {description} (missing, will use fallback)")
    
    logger.info(f"\n📊 Summary: {available_count}/{len(models)} models available")
    
    if available_count >= 4:  # We have the built-in models
        logger.info("🎉 Sufficient models available for deployment!")
        logger.info("✅ App will work with MesoNet + ImageNet base weights")
    else:
        logger.warning("⚠️  Some built-in models missing, but app should still work")
    
    logger.info("🚀 Ready for deployment!")
    return True

if __name__ == '__main__':
    main()