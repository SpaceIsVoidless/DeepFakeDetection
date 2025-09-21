web: python model_downloader.py || echo "Model download failed, continuing..."; gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --max-requests 100
