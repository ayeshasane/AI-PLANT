# AI Plant Disease Detection System

An intelligent computer vision system that uses deep learning to identify plant diseases from leaf images. This academic project demonstrates the application of Convolutional Neural Networks (CNNs) for agricultural disease detection, supporting tomato and apple plants.

## 🎯 Project Overview

This system helps farmers and agricultural professionals quickly identify plant diseases by analyzing leaf images. The AI model can distinguish between healthy plants and various disease conditions, providing accurate predictions with confidence scores.

### Key Features:
- **Multi-class Classification**: Identifies 6 different classes (3 for tomato, 3 for apple)
- **Web Interface**: User-friendly Flask web application for easy image upload
- **High Accuracy**: Uses transfer learning with MobileNetV2 pre-trained model
- **Real-time Processing**: Fast prediction on uploaded images
- **Educational Focus**: Designed as an academic project for learning AI/ML concepts

## 🌱 Supported Plants and Diseases

### Tomato Plant Diseases:
- **Healthy**: No disease symptoms
- **Early Blight**: Brown spots with concentric rings, typically on older leaves
- **Late Blight**: Water-soaked lesions that turn brown/black, rapid spread

### Apple Plant Diseases:
- **Healthy**: No disease symptoms
- **Scab**: Dark, scaly lesions on leaves and fruit
- **Black Rot**: Brown/black lesions with concentric rings, affects fruit and leaves

## 🚀 Quick Start

### Prerequisites
- Python 3.8-3.11 (TensorFlow compatible)
- 4GB+ RAM recommended
- GPU optional but recommended for training

### Installation

1. **Clone/Download the project**
   ```bash
   # Project files are already set up in your workspace
   ```

2. **Download datasets** from the specified sources and organize them in the `dataset/` folder:
   ```
   dataset/
    ├── tomato/
    │   ├── healthy/          # Tomato healthy leaf images
    │   ├── early_blight/     # Tomato early blight images
    │   └── late_blight/      # Tomato late blight images
    ├── apple/
    │   ├── healthy/          # Apple healthy leaf images
    │   ├── scab/             # Apple scab images
    │   └── black_rot/        # Apple black rot images
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**:
   ```bash
   python train.py
   ```

5. **Run the web application**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   Open your browser and go to `http://127.0.0.1:5000/`

## 📁 Project Structure

```
AI PLANT/
├── train.py              # CNN model training script
├── app.py                # Flask web application
├── demo.py               # Demonstration script
├── test_prediction.py    # Test script for batch predictions
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── dataset/             # Training images (download manually)
│   ├── tomato/
│   └── apple/
├── models/              # Saved trained model files
├── static/
│   └── uploads/         # Uploaded images for prediction
└── templates/           # HTML templates
    ├── index.html       # Upload page
    └── result.html      # Results page
```

## 🧠 Technical Details

### Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Preprocessing**: Image normalization (0-1 range)
- **Output**: 6-class multi-class classification
- **Training**: Transfer learning with fine-tuning

### Dataset Sources
- **PlantVillage Dataset**: https://github.com/spMohanty/PlantVillage-Dataset
- **Kaggle Plant Pathology**: https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7
- **Mendeley Tomato Dataset**: https://data.mendeley.com/datasets/c2x8rynybg

### Performance Metrics
- **Accuracy**: Target >90% on validation set
- **Image Processing**: Real-time preprocessing and prediction
- **Response Time**: <2 seconds per prediction

## 📊 Output Examples

### Web Application Results
When you upload a leaf image through the web interface, you'll see:

```
🌱 Plant: Tomato
🦠 Disease: Early Blight
📊 Confidence: 92.5%
```

### Command Line Testing
For batch testing with `python test_prediction.py`:

```
📷 Image: tomato_leaf.jpg
🌱 Plant: Tomato
🦠 Disease: Early Blight
📊 Confidence: 92.5%
```

### Demo Script Output
Run `python demo.py` for example predictions:

```
Plant: Tomato
Disease: Early Blight
Confidence: 92.5%
```

## 🧪 Testing the System

### Test with Sample Images
1. Place leaf images in the `test_images/` folder
2. Run: `python test_prediction.py`
3. View predictions for all images in the folder

### Web Interface Testing
1. Start the Flask app: `python app.py`
2. Open browser to `http://127.0.0.1:5000/`
3. Upload leaf images through the web form
4. View instant predictions

## 📚 How It Works

1. **Image Upload**: User uploads a leaf image via web interface or API
2. **Preprocessing**: Image is resized to 224x224 and normalized
3. **Feature Extraction**: MobileNetV2 extracts relevant features
4. **Classification**: Dense layers classify into one of 6 disease categories
5. **Output**: Returns plant type, disease name, and confidence score

## 🎓 Educational Value

This project demonstrates:
- **Computer Vision**: Image classification with CNNs
- **Transfer Learning**: Using pre-trained models for efficiency
- **Web Development**: Flask applications for AI deployment
- **Data Science**: Dataset organization and preprocessing
- **Agricultural Technology**: Real-world application of AI in farming

## ⚠️ Important Notes

- **Academic Use**: This is an educational project for learning purposes
- **Dataset Download**: Images must be downloaded manually from provided sources
- **Model Training**: Requires significant computational resources
- **Accuracy**: Results may vary based on image quality and training data
- **Not Medical Advice**: This tool is for educational purposes only

## 🤝 Contributing

This is an academic project. For improvements or modifications:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please respect the licenses of the original datasets used for training.