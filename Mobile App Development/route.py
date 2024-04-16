from flask import Flask, render_template, request
import os
import librosa
import joblib
from BabyCry_Classification import extract_mfcc

app = Flask(__name__)

# Path to the cry detection model
CRY_DETECTION_MODEL_PATH = "models/cry_detection_model.pkl"

# Load the cry detection model
cry_detection_model = joblib.load(CRY_DETECTION_MODEL_PATH)

# Function to predict cry for new audio
def predict_cry(audio_path):
    # Extract features
    features = extract_mfcc(audio_path)
    # Reshape features if necessary
    features = features.reshape(1, -1)
    # Make prediction
    prediction = cry_detection_model.predict(features)
    return prediction[0]

# Notifications data
notifications = [
    {
        "title": "Baby Needs Alert",
        "message": "Your baby might be hungry.",
        "type": "banner",
        "timestamp": "2024-04-16 12:30:00"
    },
    {
        "title": "New Message",
        "message": "You have a new message from John.",
        "type": "toast",
        "timestamp": "2024-04-16 12:25:00"
    }
]

@app.route('/')
def index():
    return render_template('index.html', notifications=notifications)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = file.filename
        file_path = os.path.join("uploads", filename)
        file.save(file_path)
        prediction = predict_cry(file_path)
        os.remove(file_path)  # Remove the uploaded file
        return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)
