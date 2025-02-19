import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/model.h5")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img

def predict_food(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction