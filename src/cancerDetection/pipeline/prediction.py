import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
import json
import base64
from io import BytesIO
from PIL import Image
from cancerDetection.utils.common import read_yaml

# Paths
MODEL_PATH = Path("artifacts/training/model_fine_tuned_best.h5")
PARAMS_PATH = Path("params.yaml")

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)
params = read_yaml(PARAMS_PATH)
image_size = tuple(params['IMAGE_SIZE'][:-1])  # e.g., (224, 224)

# Class index to label mapping
class_names = {
    0: 'adenocarcinoma',
    1: 'large_cell_carcinoma',
    2: 'normal',
    3: 'squamous_cell_carcinoma'
}

def preprocess_image(img_path):
    """Preprocess image for model prediction."""
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_from_base64(base64_string):
    """Preprocess image from base64 string for model prediction."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to PIL Image
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to target size
        img = img.resize(image_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing base64 image: {str(e)}")

def predict_image(img_path):
    """Predict class and confidence for a single image."""
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    predicted_class = class_names[predicted_idx]
    return predicted_class, confidence

def predict_image_from_base64(base64_string):
    """Predict class and confidence for a base64 encoded image."""
    img_array = preprocess_image_from_base64(base64_string)
    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    predicted_class = class_names[predicted_idx]
    return predicted_class, confidence