"""
Demo script to demonstrate plant disease detection with the trained model.
This script shows how to load an image and make predictions.
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load model and class names
def load_model_and_classes():
    try:
        model = tf.keras.models.load_model('models/plant_disease_model.h5')
        with open('models/class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        return model, class_names
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first using train.py")
        return None, None

def preprocess_image(image_path):
    """Load and preprocess an image for prediction"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_disease(image_path, model, class_names):
    """Make prediction on a leaf image"""
    processed_image = preprocess_image(image_path)
    
    if processed_image is None:
        return None
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get plant and disease names
    predicted_class = class_names[predicted_class_idx]
    plant, disease = predicted_class.split('_', 1)
    plant = plant.capitalize()
    disease = disease.replace('_', ' ').capitalize()
    
    return {
        'plant': plant,
        'disease': disease,
        'confidence': confidence,
        'all_predictions': {class_names[i]: float(predictions[0][i]) * 100 
                           for i in range(len(class_names))}
    }

def main():
    print("=" * 60)
    print("Plant Disease Detection System - Demo")
    print("=" * 60)
    
    # Load model
    model, class_names = load_model_and_classes()
    if model is None:
        return
    
    print(f"\nSupported classes: {class_names}\n")
    
    # Example usage
    print("Example Usage:")
    print("-" * 60)
    print("To detect disease in a leaf image:")
    print("  result = predict_disease('path/to/leaf.jpg', model, class_names)")
    print("\nExample prediction output:")
    example_output = {
        'plant': 'Tomato',
        'disease': 'Early Blight',
        'confidence': 92.5,
        'all_predictions': {
            'tomato_healthy': 5.2,
            'tomato_early_blight': 92.5,
            'tomato_late_blight': 2.1,
            'apple_healthy': 0.1,
            'apple_scab': 0.05,
            'apple_black_rot': 0.05
        }
    }
    
    print(f"\nPlant: {example_output['plant']}")
    print(f"Disease: {example_output['disease']}")
    print(f"Confidence: {example_output['confidence']:.1f}%")
    print("\nDetailed predictions:")
    for class_name, probability in sorted(example_output['all_predictions'].items(), 
                                         key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {probability:.2f}%")
    
    print("\n" + "=" * 60)
    print("To test with your own image:")
    print("  1. Place your leaf image in the project directory")
    print("  2. Run: result = predict_disease('your_image.jpg', model, class_names)")
    print("=" * 60)

if __name__ == '__main__':
    main()