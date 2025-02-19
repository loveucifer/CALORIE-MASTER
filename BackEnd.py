from flask import Flask, request, jsonify
import numpy as np
from preprocess import preprocess_image, model

# Define food categories and calories (example data)
FOOD_CLASSES = {0: "Pizza", 1: "Burger", 2: "Salad"}
CALORIE_INFO = {"Pizza": 285, "Burger": 354, "Salad": 152}  # Sample calorie values

app = Flask(_name_)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    image_path = "temp.jpg"
    file.save(image_path)

    prediction = model.predict(preprocess_image(image_path))
    class_index = np.argmax(prediction)
    food_name = FOOD_CLASSES[class_index]
    calories = CALORIE_INFO[food_name]

    return jsonify({"food": food_name, "calories": calories})

if _name_ == '_main_':
    app.run(debug=True)