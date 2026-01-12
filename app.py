from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

app = Flask(__name__)

# Load model and class names
model = tf.keras.models.load_model('models/plant_disease_model.h5')
with open('models/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

IMG_SIZE = (224, 224)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        
        # Preprocess and predict
        processed_image = preprocess_image(filepath)
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_class_idx]
        
        # Split to plant and disease
        plant, disease = predicted_class.split('_', 1)
        plant = plant.capitalize()
        disease = disease.replace('_', ' ').capitalize()
        
        return render_template('result.html', 
                               plant=plant, 
                               disease=disease, 
                               image_url=url_for('static', filename=f'uploads/{file.filename}'))

if __name__ == '__main__':
    app.run(debug=True)