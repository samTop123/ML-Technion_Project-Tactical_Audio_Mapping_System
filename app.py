from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
import numpy as np
import librosa
import io

app = Flask(__name__)

# on the same directory : curl -X POST -F "file=@gun-fire-346766.wav" http://127.0.0.1:5000/predict

# --- 1. Define model architecture (must match training) ---
input_shape = (431, 40)  # max_len, n_mfcc from your training
num_label_categories = 6

# --- 2. Load the trained weights ---
model = keras.models.load_model("coarse_model_saved_to_drive_version1.keras")
print("Model loaded with trained weights successfully.")

# --- 3. Load labels ---
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- 4. Preprocess audio ---
MAX_LEN = 431
N_MFCC = 40

def preprocess_audio(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfccs = mfccs.T
    # Pad or truncate to MAX_LEN
    if mfccs.shape[0] < MAX_LEN:
        padded = np.pad(mfccs, ((0, MAX_LEN - mfccs.shape[0]), (0,0)), mode='constant')
    else:
        padded = mfccs[:MAX_LEN, :]
    return padded.reshape(1, MAX_LEN, N_MFCC)

# --- 5. Flask route for predictions ---
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_bytes = request.files["file"].read()
    input_data = preprocess_audio(audio_bytes)
    predictions = model.predict(input_data)
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[0][class_index])

    return jsonify({
        "class": labels[class_index],
        "confidence": confidence
    })

# --- 6. Run server ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)