"""
Test prediction script - Use this to test the model on sample images.
Place your leaf images in the 'test_images' folder and run this script.
"""

import os
import numpy as np
from PIL import Image

# Mock model simulation since TensorFlow is not available for Python 3.14
class MockModel:
    def __init__(self):
        self.class_names = [
            'tomato_healthy', 'tomato_early_blight', 'tomato_late_blight',
            'apple_healthy', 'apple_scab', 'apple_black_rot'
        ]

    def predict(self, processed_image, verbose=0):
        # Simulate prediction with random but realistic probabilities
        np.random.seed(42)  # For consistent demo results
        probs = np.random.rand(6)
        probs = probs / probs.sum()  # Normalize to sum to 1
        return np.array([probs])

def load_model_and_classes():
    """Load the trained model and class names (mock version)"""
    try:
        # In a real scenario, this would load the actual model
        # model = tf.keras.models.load_model('models/plant_disease_model.h5')
        # with open('models/class_names.pkl', 'rb') as f:
        #     class_names = pickle.load(f)

        # Mock version for demonstration
        model = MockModel()
        class_names = model.class_names
        print("NOTE: Using mock model for demonstration (TensorFlow not available for Python 3.14)")
        print("In production, this would load the actual trained model.\n")
        return model, class_names
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Successfully loaded image: {os.path.basename(image_path)}")
        print(f"  Image size: {image.size}")
        print(f"  Image mode: {image.mode}")

        # In real implementation, this would resize and normalize
        # image = image.resize((224, 224))
        # image = np.array(image) / 255.0
        # image = np.expand_dims(image, axis=0)

        return image  # Return PIL image for mock
    except Exception as e:
        print(f"✗ Error processing {image_path}: {e}")
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
    print(f"Found {len(image_files)} image(s) to analyze:\n")

    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        result = predict_disease(image_path, model, class_names)

        if result:
            print(f"📷 Image: {result['image']}")
            print(f"🌱 Plant: {result['plant']}")
            print(f"🦠 Disease: {result['disease']}")
            print(f"📊 Confidence: {result['confidence']:.2f}%")
            print("📈 All predictions:")
            for class_name, prob in sorted(result['predictions'].items(),
                                         key=lambda x: x[1], reverse=True):
                plant_name, disease_name = class_name.split('_', 1)
                plant_name = plant_name.capitalize()
                disease_name = disease_name.replace('_', ' ').capitalize()
                marker = "✓" if prob == max(result['predictions'].values()) else " "
                print(f"    {marker} {plant_name} - {disease_name}: {prob:.2f}%")
            print("-" * 50)

    print("\n" + "=" * 70)
    print("NOTE: These are simulated results for demonstration purposes.")
    print("To get real predictions:")
    print("1. Use Python 3.8-3.11 (TensorFlow compatible)")
    print("2. Install: pip install tensorflow flask pillow numpy matplotlib scikit-learn")
    print("3. Download datasets and run: python train.py")
    print("=" * 70)

if __name__ == '__main__':
    main()