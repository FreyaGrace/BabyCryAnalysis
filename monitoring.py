from flask import Flask, Response, render_template, jsonify
from threading import Thread
import sounddevice as sd
from scipy.io.wavfile import read
import numpy as np
import pickle
import librosa
from io import BytesIO
import time

# Assuming scikit-learn model is saved as a joblib file
model = pickle.load(open(r'C:\Users\acer\Desktop\Cryanalysis\myRandomForest.pkl', 'rb'))  # Replace with your model filename


# Replace with your desired audio stream configuration
audio_stream_config = {
    "samplerate": 44100,
    "channels": 1,  # Assuming monophonic audio
}

# Flag to control sound recording
recording = False
audio_buffer = []  # Buffer to store audio samples temporarily
cry_detected = False  # Flag to indicate cry detection
cry_threshold = 0.8  # Threshold for cry probability (adjust as needed)
BUFFER_DURATION = 5  # Example buffer duration in seconds

def record_audio():
    global recording, audio_buffer, cry_detected
    with sd.InputStream(**audio_stream_config) as stream:
        while recording:
            data = stream.read(1024)  # Adjust buffer size as needed
            audio_buffer.append(data.tobytes())

            # Check for silence or inactivity (optional)
            # ...

            # Process buffer if a certain duration or threshold is met
            if len(audio_buffer) >= BUFFER_DURATION * audio_stream_config["samplerate"]:
                cry_probability = predict_cry(b''.join(audio_buffer))
                cry_detected = cry_probability >= cry_threshold
                # Clear buffer after processing
                audio_buffer.clear()

def predict_cry(audio_data):
    # Extract MFCC features from audio data
    mfcc_features = extract_mfcc(audio_data)  # Replace with your feature extraction function

    # Make predictions using the model
    prediction = model.predict_proba([mfcc_features])[:, 1]  # Get probability for cry class
    return prediction[0]

def extract_mfcc(audio_file, max_length=100):
    audiofile, sr = librosa.load(BytesIO(audio_file))
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=20)
    if fingerprint.shape[1] < max_length:
        pad_width = max_length - fingerprint.shape[1]
        fingerprint_padded = np.pad(fingerprint, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return fingerprint_padded.T
    elif fingerprint.shape[1] > max_length:
        return fingerprint[:, :max_length].T
    else:
        return fingerprint.T

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def stream_audio():
    global audio_buffer
    while True:
        if len(audio_buffer) > 0:
            frame = audio_buffer.pop(0)
            yield (b'--frame\r\n'
                   b'Content-Type: audio/wav\r\n\r\n' + frame + b'\r\n')
        else:
            # Simulate a small delay if no data is available
            yield (b'--frame\r\n'
                   b'Content-Type: audio/wav\r\n\r\n' + b'\r\n')
            time.sleep(0.1)  # Adjust delay as needed

@app.route("/audio_stream")
def audio_stream():
    return Response(stream_audio(), mimetype="audio/x-wav", access_control_allow_origin='*')

@app.route("/start_recording")
def start_recording():
    global recording, cry_detected
    recording = True
    audio_buffer.clear()
    cry_detected = False
    recording_thread = Thread(target=record_audio)
    recording_thread.start()
    return "Recording started"

@app.route("/stop_recording")
def stop_recording():
    global recording
    recording = False
    return "Recording stopped"

@app.route("/is_cry_detected")
def is_cry_detected():
    global cry_detected
    return jsonify({"cry_detected": cry_detected})

if __name__ == "__main__":
    app.run(debug=True)
