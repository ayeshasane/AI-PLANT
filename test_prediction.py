"""
Test prediction script - Use this to test the model on sample images.
Place your leaf images in the 'test_images' folder and run this script.
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

def load_model_and_classes():
    """Load the trained model and class names"""
    try:
        model = tf.keras.models.load_model('models/plant_disease_model.h5')
        with open('models/class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        return model, class_names
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python train.py")
        return None, None

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def predict_disease(image_path, model, class_names):
    """Predict disease for a leaf image"""
    processed_image = preprocess_image(image_path)
    
    if processed_image is None:
        return None
    
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[0][predicted_class_idx] * 100
    
    predicted_class = class_names[predicted_class_idx]
    plant, disease = predicted_class.split('_', 1)
    
    return {
        'image': os.path.basename(image_path),
        'plant': plant.capitalize(),
        'disease': disease.replace('_', ' ').capitalize(),
        'confidence': confidence,
        'predictions': {class_names[i]: float(predictions[0][i]) * 100 
                       for i in range(len(class_names))}
    }

def main():
    # Load model
    model, class_names = load_model_and_classes()
    if model is None:
        return
    
    # Create test_images directory if it doesn't exist
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created '{test_dir}' folder. Place your leaf images here and run again.")
        return
    
    # Find all images in test_images folder
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in '{test_dir}' folder.")
        print("Please add leaf images (.jpg, .png) to the test_images folder.")
        return
    
    print("\n" + "=" * 70)
    print("Plant Disease Detection - Test Results")
    print("=" * 70)
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        result = predict_disease(image_path, model, class_names)
        
        if result:
            print(f"\nImage: {result['image']}")
            print(f"  Plant: {result['plant']}")
            print(f"  Disease: {result['disease']}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            print("  All predictions:")
            for class_name, prob in sorted(result['predictions'].items(), 
                                         key=lambda x: x[1], reverse=True):
                print(f"    {class_name}: {prob:.2f}%")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()