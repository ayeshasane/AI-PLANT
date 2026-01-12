import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Adjust as needed
DATASET_PATH = 'dataset/'

# Function to load image paths and labels
def load_data_paths(dataset_path):
    image_paths = []
    labels = []
    class_names = []
    
    for plant in os.listdir(dataset_path):
        plant_path = os.path.join(dataset_path, plant)
        if os.path.isdir(plant_path):
            for disease in os.listdir(plant_path):
                disease_path = os.path.join(plant_path, disease)
                if os.path.isdir(disease_path):
                    class_name = f"{plant}_{disease}"
                    if class_name not in class_names:
                        class_names.append(class_name)
                    for img_file in os.listdir(disease_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(disease_path, img_file))
                            labels.append(class_name)
    
    return image_paths, labels, class_names

# Load data
image_paths, labels, class_names = load_data_paths(DATASET_PATH)

# Create label to index mapping
label_to_index = {label: idx for idx, label in enumerate(class_names)}
index_to_label = {idx: label for idx, label in enumerate(class_names)}

# Convert labels to indices
label_indices = [label_to_index[label] for label in labels]

# Create TensorFlow dataset
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_indices))
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=len(image_paths))
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Split into train and validation (80-20)
train_size = int(0.8 * len(image_paths))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Build model using MobileNetV2
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# Save the model
model.save('models/plant_disease_model.h5')

# Save class names
with open('models/class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

print("Model trained and saved successfully.")
print(f"Class names: {class_names}")