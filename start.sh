#!/bin/bash

# Startup script for DeepFake Detection App
# This script handles model downloading and app startup

set -e  # Exit on any error

echo "🚀 Starting DeepFake Detection App..."

# Check Python version
python_version=$(python --version 2>&1)
echo "📍 Using $python_version"

# Download models if needed
echo "📦 Checking model files..."
if python model_downloader.py; then
    echo "✅ All models ready"
else
    echo "⚠️  Some models failed to download, continuing with available models..."
fi

# Check available models
echo "🔍 Checking model availability..."
python -c "
import os
models = ['Meso4_DF.h5', 'tf_model.h5', 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5']
for model in models:
    status = '✅' if os.path.exists(model) else '❌'
    size = f'({os.path.getsize(model)/1024/1024:.1f}MB)' if os.path.exists(model) else ''
    print(f'{status} {model} {size}')
"

# Set production environment
export FLASK_ENV=production

# Start the application
echo "🌐 Starting Flask application..."
if [ "$1" = "dev" ]; then
    echo "🔧 Development mode"
    python app.py
else
    echo "🚀 Production mode"
    exec gunicorn app:app \
        --bind 0.0.0.0:${PORT:-8080} \
        --workers ${WORKERS:-1} \
        --timeout ${TIMEOUT:-120} \
        --max-requests ${MAX_REQUESTS:-100} \
        --access-logfile - \
        --error-logfile -
fi