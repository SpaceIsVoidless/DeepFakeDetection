# 🕵️ DeepFake Detection App

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

**AI-Powered Deepfake Detection with Multiple Neural Networks**

[🚀 Live Demo](https://your-app.onrender.com) • [📖 Documentation](DEPLOYMENT.md) • [🐛 Report Bug](https://github.com/yourusername/deepfake-detection-app/issues)

</div>

## ✨ Features

- 🧠 **Multiple AI Models**: MesoNet, ResNet50, Xception for comprehensive analysis
- 🖼️ **Image & Video Support**: Analyze photos, selfies, and video files
- 🔍 **Artifact Detection**: Advanced computer vision to detect deepfake artifacts
- ⚡ **Real-time Processing**: Fast analysis with intelligent calibration
- 🎯 **High Accuracy**: Optimized to correctly identify both real photos and deepfakes
- 🛡️ **Production Ready**: Security features, monitoring, and error handling
- 📱 **Modern UI**: Responsive design with dark/light themes

## 🎯 How It Works

The app uses a **multi-model ensemble approach** with intelligent calibration:

1. **Image Analysis**: Detects compression artifacts, color inconsistencies, and edge anomalies
2. **Model Inference**: Runs multiple neural networks trained on deepfake datasets
3. **Smart Calibration**: Combines model predictions with artifact analysis
4. **Confidence Scoring**: Provides detailed confidence metrics for each prediction

## 🚀 Quick Start

### Option 1: Deploy to Render.com (Recommended)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Click the deploy button above
2. Connect your GitHub repository
3. Wait for automatic model download (~10 minutes)
4. Your app will be live! 🎉

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-detection-app.git
cd deepfake-detection-app

# Install dependencies
pip install -r requirements.txt

# Download large model files
python model_downloader.py

# Run the application
python app.py
```

Visit `http://localhost:5000` to use the app!

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API      │    │   AI Models     │
│                 │    │                  │    │                 │
│ • File Upload   │───▶│ • File Validation│───▶│ • MesoNet-4     │
│ • Results UI    │    │ • Image Processing│    │ • ResNet50      │
│ • Progress Bar  │    │ • Model Ensemble │    │ • Xception      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Artifact Detection│
                       │                  │
                       │ • Compression    │
                       │ • Color Analysis │
                       │ • Edge Detection │
                       └──────────────────┘
```

## 📊 Model Performance

| Model | Accuracy | Speed | Specialization |
|-------|----------|-------|----------------|
| **MesoNet-4** | 🟢 High | ⚡ Fast | Face-focused detection |
| **ResNet50** | 🟡 Medium | 🐌 Slow | General image analysis |
| **Xception** | 🟢 High | 🐌 Slow | Feature extraction |
| **Ensemble** | 🟢 **Best** | ⚡ Optimized | Combined intelligence |

## 🛠️ Technology Stack

- **Backend**: Python 3.11, Flask 3.0
- **AI/ML**: TensorFlow 2.15, OpenCV, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Docker, Render.com, Railway
- **Models**: Custom trained neural networks

## 📁 Project Structure

```
deepfake-detection-app/
├── 🐍 app.py                 # Main Flask application
├── 🤖 model_downloader.py    # Automatic model downloading
├── 📋 requirements.txt       # Python dependencies
├── 🎨 templates/
│   └── index.html           # Web interface
├── 🧠 models/
│   ├── classifiers.py       # Neural network definitions
│   └── *.h5                # Pre-trained model weights
├── 🚀 render.yaml           # Deployment configuration
├── 🐳 Dockerfile            # Container configuration
└── 📖 README.md             # This file
```

## 🔧 Configuration

### Environment Variables

```bash
FLASK_ENV=production          # Production mode
PORT=8080                     # Server port
MAX_CONTENT_LENGTH=33554432   # 32MB upload limit
```

### Model Files

The app automatically downloads large model files at startup:

- ✅ **Small models** (included in repo): `Meso4_*.h5` (~0.15MB each)
- ⬇️ **Large models** (auto-downloaded): `tf_model.h5` (98MB), `xception_*.h5` (80MB)

## 🚀 Deployment Options

### 🥇 Render.com (Recommended)
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Perfect for ML apps

### 🥈 Railway.app
- ✅ Simple deployment
- ✅ Usage-based pricing

### 🥉 Docker
```bash
docker build -t deepfake-app .
docker run -p 8080:8080 deepfake-app
```

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/upload` | POST | Analyze image/video |
| `/health` | GET | Health check |
| `/models` | GET | Model status |

### Example API Usage

```bash
# Health check
curl https://your-app.onrender.com/health

# Upload file for analysis
curl -X POST -F "file=@image.jpg" https://your-app.onrender.com/upload
```

## 🛡️ Security Features

- 🔒 **File Validation**: Strict file type and size checking
- 🧹 **Automatic Cleanup**: Temporary files are automatically removed
- 🛡️ **Input Sanitization**: All inputs are validated and sanitized
- 📊 **Rate Limiting**: Built-in protection against abuse
- 🔍 **Security Headers**: Production-ready security configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MesoNet](https://github.com/DariusAf/MesoNet) - Original MesoNet implementation
- [FaceForensics++](https://github.com/ondyari/FaceForensics) - Dataset and models
- [TensorFlow](https://tensorflow.org) - Machine learning framework

## 📞 Support

- 📧 **Email**: your.email@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/deepfake-detection-app/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/deepfake-detection-app/discussions)

---

<div align="center">

**Made with ❤️ for AI Safety**

[⭐ Star this repo](https://github.com/yourusername/deepfake-detection-app) if you found it helpful!

</div>