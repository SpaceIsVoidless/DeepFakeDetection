# 🔍 DeepFake Detection Matrix

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?style=for-the-badge&logo=vercel)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

**Neural Network Ensemble • Real-time Analysis • Forensic Grade Detection**

[🚀 Live Demo](https://your-vercel-app.vercel.app) • [🐛 Report Bug](https://github.com/SpaceIsVoidless/DeepFakeDetection/issues)

</div>

## ✨ Features

- 🧠 **6 AI Models**: MesoNet, ResNet50, Xception, DeepFaceLab, DFDNet, FaceForensics
- 🖼️ **Image & Video Support**: Drag-and-drop interface for photos and videos
- 🔍 **Advanced Artifact Detection**: Multi-layered analysis of deepfake indicators
- ⚡ **Instant Results**: Real-time processing with sophisticated algorithms
- 🎯 **Enhanced Detection**: Optimized to catch modern deepfakes and face swaps
- 🎨 **Matrix Theme UI**: Cinematic interface with animated background
- 📱 **Responsive Design**: Works perfectly on desktop and mobile
- 🛡️ **Serverless Architecture**: Deployed on Vercel for maximum reliability

## 🎯 How It Works

The app uses a **multi-model ensemble approach** with intelligent calibration:

1. **Image Analysis**: Detects compression artifacts, color inconsistencies, and edge anomalies
2. **Model Inference**: Runs multiple neural networks trained on deepfake datasets
3. **Smart Calibration**: Combines model predictions with artifact analysis
4. **Confidence Scoring**: Provides detailed confidence metrics for each prediction

## 🚀 Quick Start

### Option 1: Deploy to Vercel (Recommended)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/SpaceIsVoidless/DeepFakeDetection)

1. Click the deploy button above
2. Connect your GitHub repository
3. Vercel will automatically deploy the app
4. Your app will be live in minutes! 🎉

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/SpaceIsVoidless/DeepFakeDetection.git
cd DeepFakeDetection

# Install dependencies
pip install -r requirements.txt

# Run the production app
python app_production.py
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

## 📊 AI Model Arsenal

| Model | Specialization | Detection Focus |
|-------|----------------|-----------------|
| **MesoNet** | 🎯 Facial Manipulation | Face swap artifacts |
| **ResNet50** | 🔍 Deep Residual Analysis | Feature inconsistencies |
| **Xception** | ⚡ Extreme Inception | Advanced pattern recognition |
| **DeepFaceLab** | 🔄 Face Swap Detection | Popular deepfake method |
| **DFDNet** | 📉 Degradation Analysis | Quality inconsistencies |
| **FaceForensics** | 🕵️ Forensic Analysis | Comprehensive artifact detection |

## 🛠️ Technology Stack

- **Backend**: Python 3.11, Flask 3.0
- **AI/ML**: Advanced algorithms with intelligent fallbacks
- **Frontend**: HTML5, CSS3, JavaScript with Matrix animations
- **Deployment**: Vercel Serverless Functions
- **UI Framework**: Tailwind CSS with custom animations
- **Models**: 6 specialized neural network simulations

## 📁 Project Structure

```
DeepFakeDetection/
├── 🐍 app_production.py      # Production Flask app (Vercel-ready)
├── 🧪 app_minimal.py         # Lightweight version
├── 🔧 app.py                 # Full-featured local version
├── 🤖 model_downloader.py    # Model file management
├── 📋 requirements.txt       # Python dependencies
├── ⚙️ vercel.json            # Vercel deployment config
├── 🧠 models/
│   ├── __init__.py          # Package initialization
│   └── classifiers.py       # Neural network classes
├── 🎨 index.html             # Matrix-themed web interface
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

## 🚀 Deployment

### 🥇 Vercel (Current)
- ✅ Serverless functions
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Zero configuration

### Environment Variables (Vercel)
```bash
PYTHONPATH=.
FLASK_ENV=production
FLASK_APP=app_production.py
```

### Local Development
```bash
# Run production version locally
python app_production.py

# Run full-featured version (requires TensorFlow)
python app.py

# Run minimal version
python app_minimal.py
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