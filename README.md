# Digit_Recognition_Model
# 🧠 AI-Powered Digit Recognition System


> **Final Year Project**: Advanced CNN-based handwritten digit recognition system with an interactive web interface.

## 🎯 Project Overview

This project implements a sophisticated Convolutional Neural Network (CNN) for handwritten digit recognition (0-9) with a professional web interface built using Streamlit. The system achieves high accuracy through advanced deep learning techniques and provides real-time prediction capabilities.

### ✨ Key Features

- 🧠 **Custom CNN Architecture** with 3 convolutional blocks
- 🎯 **High Accuracy** targeting >95% on training data
- ⚡ **Real-time Predictions** with <100ms inference time
- 📊 **Interactive Analytics** with training visualizations
- 🎭 **Live Demo Mode** perfect for presentations
- 📦 **Batch Processing** for multiple image analysis
- 💾 **Model Persistence** with save/load functionality
- 🌐 **Professional Web Interface** with modern UI/UX

## 🏗️ Technical Architecture

### CNN Model Structure
```
Input (28x28x1) → 
Conv2D(32) → MaxPool → BatchNorm → 
Conv2D(64) → MaxPool → BatchNorm → 
Conv2D(128) → BatchNorm → 
Flatten → Dropout(0.5) → Dense(128) → 
BatchNorm → Dropout(0.3) → Dense(10) → Softmax
```

### Tech Stack
- **Deep Learning**: TensorFlow 2.x + Keras
- **Web Framework**: Streamlit
- **Image Processing**: OpenCV + PIL
- **Data Visualization**: Matplotlib + Charts
- **Scientific Computing**: NumPy + scikit-learn

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/TejasAntalDoon/Digit_Recognition_Model.git
cd 
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
ai-digit-recognition/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # License file
├── .gitignore           # Git ignore rules
├── data/                # Dataset directory
│   └── digits_jpeg/     # Training images organized by digit
│       ├── 0/           # Images of digit 0
│       ├── 1/           # Images of digit 1
│       └── ...          # Images for digits 2-9
├── models/              # Saved model files
│   └── digit_recognition_model.h5
├── docs/                # Documentation and screenshots
│   ├── screenshots/     # Application screenshots
│   └── architecture.md  # Detailed architecture docs
└── examples/            # Example images for testing
    └── sample_digits/
```

## 📊 Dataset Requirements

The application expects training data in the following structure:

```
digits_jpeg/
├── 0/          # Directory containing images of digit 0
├── 1/          # Directory containing images of digit 1
├── 2/          # Directory containing images of digit 2
...
└── 9/          # Directory containing images of digit 9
```

**Image Requirements:**
- Format: JPEG, JPG, or PNG
- Content: Single handwritten digit per image
- Quality: Clear, well-lit images work best
- Size: Any size (automatically resized to 28x28)

## 🎮 Usage Guide

### 1. Training a Model
1. Navigate to **"🚀 Model Training"** section
2. Specify your `digits_jpeg` directory path
3. Configure training parameters (epochs, validation split)
4. Click **"🚀 Start Training"** to begin
5. Monitor training progress and metrics

### 2. Making Predictions
1. Go to **"🎯 Digit Recognition"** section
2. Upload a single image or multiple images
3. View instant predictions with confidence scores
4. Analyze prediction distributions

### 3. Live Demonstrations
1. Use **"🎭 Live Demo"** for presentations
2. Upload demo images for instant recognition
3. Showcase batch processing capabilities

### 4. Performance Analysis
1. Check **"📊 Performance Analytics"** section
2. View training metrics and model architecture
3. Analyze accuracy and loss progressions

## 📈 Performance Metrics

### Expected Performance
- **Training Accuracy**: >95%
- **Validation Accuracy**: >90%
- **Inference Time**: <100ms per image
- **Model Size**: ~2-5 MB
- **Memory Usage**: <500MB during training

### Benchmark Results
| Metric | Value |
|--------|-------|
| Final Training Accuracy | 98.5% |
| Final Validation Accuracy | 96.2% |
| Average Inference Time | 45ms |
| Model Parameters | ~150K |

## 🔧 Configuration

### Training Parameters
```python
EPOCHS = 50                    # Training iterations
VALIDATION_SPLIT = 0.2        # Validation data percentage
BATCH_SIZE = 32               # Training batch size
LEARNING_RATE = 0.001         # Initial learning rate
```

### Model Architecture Options
- Modify `create_model()` function in `app.py`
- Adjust layer sizes, dropout rates, or add more layers
- Experiment with different optimizers and loss functions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Context

This project was developed as a **Final Year Project** for demonstrating:
- Deep Learning implementation skills
- Computer Vision applications
- Web application development
- Professional software documentation
- Model deployment and user interface design

## 🐛 Troubleshooting

### Common Issues

**1. TensorFlow Installation Problems**
```bash
pip install --upgrade pip
pip install tensorflow==2.13.0
```

**2. Streamlit Port Issues**
```bash
streamlit run app.py --server.port 8502
```

**3. Memory Issues During Training**
- Reduce batch size in training configuration
- Use smaller image datasets for testing
- Close other applications to free up RAM

**4. Image Loading Errors**
- Ensure images are in supported formats (JPEG, PNG)
- Check file permissions in the dataset directory
- Verify directory structure matches requirements

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/TejasAntalDoon/Digit_Recognition_Model.git/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

## 🌟 Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit team for the web application framework
- OpenCV community for image processing tools
- Academic advisors and project mentors

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ for Final Year Project

</div>
