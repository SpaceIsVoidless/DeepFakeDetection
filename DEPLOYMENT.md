# DeepFake Detection App - Deployment Guide

## Overview

This guide covers deploying the DeepFake Detection webapp to various cloud platforms. The app uses both small model files (included in repo) and large model files (downloaded at startup).

## Model Files Strategy

### Small Models (Included in Repository)
- `Meso4_DF.h5` (~0.15 MB) ✅
- `Meso4_F2F.h5` (~0.15 MB) ✅  
- `MesoInception_DF.h5` (~0.19 MB) ✅
- `MesoInception_F2F.h5` (~0.19 MB) ✅

### Large Models (Downloaded at Startup)
- `tf_model.h5` (~98 MB) ⬇️ Downloaded
- `xception_weights_tf_dim_ordering_tf_kernels_notop.h5` (~80 MB) ⬇️ Downloaded

## Deployment Platforms

### 1. Render.com (Recommended)

**Why Render?**
- Supports large model downloads
- Good for ML applications
- Persistent storage
- Reasonable free tier

**Steps:**
1. Push code to GitHub (large .h5 files will be ignored)
2. Connect Render to your GitHub repo
3. Configure build settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python model_downloader.py && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
4. Set environment variables:
   - `PYTHON_VERSION`: `3.11.5`
   - `FLASK_ENV`: `production`

### 2. Railway.app

**Steps:**
1. Connect Railway to your GitHub repo
2. Railway will automatically detect the Python app
3. The Procfile will handle model downloading
4. Set environment variables:
   - `FLASK_ENV`: `production`

### 3. Heroku

**Steps:**
1. Install Heroku CLI
2. Create Heroku app: `heroku create your-app-name`
3. Push to Heroku: `git push heroku main`
4. The Procfile will handle model downloading
5. Set config vars:
   ```bash
   heroku config:set FLASK_ENV=production
   ```

### 4. Google Cloud Run

**Steps:**
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   # Download models at build time
   RUN python model_downloader.py
   
   EXPOSE 8080
   CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]
   ```

2. Build and deploy:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/deepfake-app
   gcloud run deploy --image gcr.io/PROJECT-ID/deepfake-app --platform managed
   ```

## Environment Variables

Set these environment variables in your deployment platform:

```bash
FLASK_ENV=production          # Disable debug mode
PORT=8080                     # Port (usually auto-set)
MAX_CONTENT_LENGTH=33554432   # 32MB file upload limit
```

## Build Process

The deployment process follows these steps:

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Download Models**: `python model_downloader.py`
3. **Start Application**: `gunicorn app:app`

## Troubleshooting

### Model Download Issues

If models fail to download:

1. **Check logs** for download errors
2. **Verify URLs** in `model_downloader.py`
3. **Manual download**: Run `python model_downloader.py --force`
4. **Fallback**: App will work with available models

### Memory Issues

If deployment fails due to memory:

1. **Reduce workers**: Use `--workers 1` in Procfile
2. **Increase timeout**: Use `--timeout 120`
3. **Choose platform** with more memory (Render > Heroku free tier)

### Build Timeout

If build times out during model download:

1. **Pre-download models** and include in repo (if platform allows large files)
2. **Use Docker** with multi-stage builds
3. **Download models** in background after startup

## Performance Optimization

### For Production:

1. **Enable model caching**:
   ```python
   # In app.py, models are loaded once at startup
   ```

2. **Use CDN** for static assets:
   ```python
   # Add CDN URLs for CSS/JS in templates
   ```

3. **Add Redis caching** (optional):
   ```bash
   pip install redis flask-caching
   ```

## Security Considerations

1. **File upload limits** are enforced (32MB)
2. **Filename sanitization** prevents path traversal
3. **Content validation** ensures only valid files are processed
4. **Temporary file cleanup** prevents disk space issues

## Monitoring

The app includes several monitoring endpoints:

- `/health` - Basic health check with model status
- `/models` - Detailed model information
- Processing time tracking in responses

## Cost Estimation

### Render.com:
- **Free tier**: 750 hours/month
- **Paid tier**: $7/month for better performance

### Railway.app:
- **Free tier**: $5 credit/month
- **Usage-based**: ~$0.000463/GB-hour

### Heroku:
- **Free tier**: Discontinued
- **Basic**: $7/month

## Support

For deployment issues:

1. Check the logs for specific error messages
2. Verify all model files downloaded successfully
3. Test locally with `FLASK_ENV=production`
4. Check platform-specific documentation

## Quick Start Commands

```bash
# Test model download locally
python model_downloader.py

# Test production mode locally
FLASK_ENV=production python app.py

# Check model status
curl http://localhost:5000/health
```