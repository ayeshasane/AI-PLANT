# AI Plant Disease Detection System

This project uses a Convolutional Neural Network (CNN) to detect diseases in tomato and apple leaves from uploaded images.

## Supported Plants and Diseases

- **Tomato**: Healthy, Early Blight, Late Blight
- **Apple**: Healthy, Scab, Black Rot

## Setup

1. Download the datasets from the specified sources and organize them in the `dataset/` folder as follows:
   ```
   dataset/
    ├── tomato/
    │   ├── healthy/
    │   ├── early_blight/
    │   └── late_blight/
    ├── apple/
    │   ├── healthy/
    │   ├── scab/
    │   └── black_rot/
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Train the model:
   ```
   python train.py
   ```

4. Run the Flask app:
   ```
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000/` to upload images and get predictions.

## Project Structure

- `train.py`: Script to train the CNN model
- `app.py`: Flask web application for image upload and prediction
- `models/`: Directory to save the trained model and class names
- `static/uploads/`: Directory for uploaded images
- `templates/`: HTML templates for the web interface
- `dataset/`: Directory for training images (to be downloaded manually)

## Model Details

- Uses MobileNetV2 as base model with transfer learning
- Image size: 224x224
- Normalized pixel values (0-1)
- Multi-class classification with 6 classes