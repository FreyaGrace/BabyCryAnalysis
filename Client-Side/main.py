import wave
import joblib
import pyaudio as pa
import librosa
import matplotlib.pyplot as plt
import uuid
import struct
import numpy as np
import os
import tempfile  # Import tempfile module

# Parameters for audio recording
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FORMAT = pa.paInt16
RECORD_DURATION = 10  # Duration to record after cry is detected
MODEL_FILE = r"C:\Users\acer\Documents\Cry\cry\cry_detection.pkl"  # Path to your trained cry detection model
CLASSIFICATION_FILE = r"C:\Users\acer\Documents\Cry\cry\models\myRandomForest.pkl"


def chop_new_audio(audio_data, folder):
    os.makedirs(folder, exist_ok=True)  # Create directory if it doesn't exist
    audio = wave.open(audio_data, 'rb')
    frame_rate = audio.getframerate()
    n_frames = audio.getnframes()
    window_size = 2 * frame_rate
    num_secs = int(np.ceil(n_frames / frame_rate))
    last_number_frames = 0
    for i in range(num_secs):
        shortfilename = str(uuid.uuid4())  # Generate a unique filename
        snippetfilename = f"{folder}/{shortfilename}snippet{i+1}.wav"
        snippet = wave.open(snippetfilename, 'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(audio.getsampwidth())
        snippet.setframerate(frame_rate)
        snippet.setnframes(audio.getnframes())
        snippet.writeframes(audio.readframes(window_size))
        audio.setpos(audio.tell() - 1 * frame_rate)

        # Check if the frame size of the snippet matches the previous snippets
        if last_number_frames < 1:
            last_number_frames = snippet.getnframes()
        elif snippet.getnframes() != last_number_frames:
            os.rename(snippetfilename, f"{snippetfilename}.bak")
        snippet.close()

def extract_features(audio_file, max_length=100):
    audiofile, sr = librosa.load(audio_file)
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=20)
    if fingerprint.shape[1] < max_length:
        pad_width = max_length - fingerprint.shape[1]
        fingerprint_padded = np.pad(fingerprint, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return fingerprint_padded.T
    elif fingerprint.shape[1] > max_length:
        return fingerprint[:, :max_length].T
    else:
        return fingerprint.T

def predict(audio_data, sr, model):
    """Predicts whether the audio contains a cry."""
    features = extract_features(audio_data)
    prediction = model.predict(features)
    if prediction[0] == 1:
        print("Cry detected!")
    else:
        print("No cry detected.")

def normalize_audio(audio_data):
    """Normalizes audio data by dividing by the maximum absolute value."""
    return audio_data / np.max(np.abs(audio_data))

def classify_cry(audio_path, classification_model_file=CLASSIFICATION_FILE):
    """Classifies the recorded audio to determine if it's a cry."""
    try:
        # Load the trained classification model
        model = joblib.load(classification_model_file)

        # Predict on new audio snippets
        predictions = []
        folder_path = audio_path
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                filepath = os.path.join(folder_path, filename)
                audiofile, sr = librosa.load(filepath, sr=RATE)  # Load audio file
                fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=20)  # Extract features
                fingerprint_flat = fingerprint.reshape(-1)  # Flatten the MFCC features
                # Pad or truncate features to match the number of features used for training
                if len(fingerprint_flat) < 2000:
                    fingerprint_flat = np.pad(fingerprint_flat, (0, 2000 - len(fingerprint_flat)))
                elif len(fingerprint_flat) > 2000:
                    fingerprint_flat = fingerprint_flat[:2000]
                prediction = model.predict([fingerprint_flat])  # Reshape to match expected input format
                predictions.append(prediction[0])

        from collections import Counter
        data = Counter(predictions)
        print(data.most_common())  # Returns all unique items and their counts
        print(data.most_common(1))  # Returns the most common prediction

    except Exception as e:
        print(f"Error classifying cry audio:", e)

def record_audio(filename, duration):
    """Records audio for a specified duration."""
    p = pa.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    wf = wave.open(filename, 'wb')  # Use the provided filename
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    for _ in range(int(duration * RATE / CHUNK)):
        data = stream.read(CHUNK)
        wf.writeframes(data)

    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()


def monitor():
    """Continuously monitors and processes audio."""
    try:
        # Load the trained model
        model = joblib.load(MODEL_FILE)

        # Flag to control monitoring state
        monitoring = True

        prediction = None  # Initialize prediction variable
        
        while monitoring:
            p = pa.PyAudio()
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            output=True,  # No need for output here
                            frames_per_buffer=CHUNK)

            fig, ax = plt.subplots()
            x = np.arange(0, 2 * CHUNK, 2)
            line, = ax.plot(x, np.random.rand(CHUNK), 'r')
            ax.set_ylim(-32768, 32768)  # Adjust y-axis limits based on audio format
            ax.set_xlim(0, CHUNK)
            plt.show(block=False)

            # Flag to control recording state
            recording = False

            while True:
                # Read audio chunk
                data = stream.read(CHUNK)
                if isinstance(data, bytearray):
                    data = bytes(data)
                    data_array = np.frombuffer(data, dtype=np.int16)
                    # Create a temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix=".wav", mode='wb') as temp_wav:
                        temp_wav.write(data.tobytes())  # Assuming data is in bytes format
                        prediction = predict(temp_wav.name, RATE, model)

                    # Update the plot with the received audio data
                    line.set_ydata(data_array)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                if prediction == 1 and not recording:
                    print("Crying detected! Recording for classification...")
                    recording = True
                    # Record audio for RECORD_DURATION seconds
                    record_audio("cry.wav", RECORD_DURATION)
                    # Classify the recorded audio
                    classify_cry("cry.wav")
                    # Stop monitoring temporarily
                    monitoring = False
                    break  # Break out of the inner loop to stop monitoring temporarily
                elif prediction == 0 and recording:
                    # Reset recording flag if crying stops
                    recording = False

            # Close audio stream
            stream.stop_stream()
            stream.close()
            p.terminate()

    except Exception as e:
        print(f"Error processing audio:", e)

if __name__ == "__main__":
    monitor()
