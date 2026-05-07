import io
import time
import secrets

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import tensorflow as tf
import numpy as np
import librosa

# ==========================================
# --- CONFIGURATION & SETUP ---
# ==========================================

# app: The Flask application instance that handles web routing and requests.
app = Flask(__name__)
# app.secret_key: A secure random string used to encrypt session cookies for user authentication.
app.secret_key = secrets.token_hex(16)  

# MAX_LEN: The fixed temporal length (number of frames) required for the model input.
MAX_LEN = 431
# N_MFCC: The number of Mel-frequency cepstral coefficients to extract from each audio frame.
N_MFCC = 40
# INPUT_SHAPE: A tuple defining the expected matrix dimensions for the CNN input.
INPUT_SHAPE = (MAX_LEN, N_MFCC)

# USERS: A dictionary serving as a mock database for user credentials and role-based access.
USERS = {
    "soldier1": {"password": "123", "role": "soldier"},
    "soldier2": {"password": "123", "role": "soldier"},
    "commander1": {"password": "admin", "role": "commander"}
}

# live_field_status: A global shared dictionary storing the most recent detection status for each soldier.
live_field_status = {}


# ==========================================
# --- MODEL LOADING ---
# ==========================================

print("Loading model...")
# model: The loaded Keras/TensorFlow CNN model used for classifying audio features.
model = tf.keras.models.load_model("model_train_clean_cnn.keras")
print("Model loaded successfully.")

# f: The file object used for reading the class labels from the local filesystem.
with open("labels.txt", "r") as f:
    # labels: A list of strings where each entry corresponds to a specific sound class (e.g., 'Gunfire').
    labels = [line.strip() for line in f.readlines()]


# ==========================================
# --- HELPER FUNCTIONS ---
# ==========================================

def preprocess_audio(audio_bytes):
    """Converts raw audio bytes into padded MFCC features for the model."""
    # y: A 1D numpy array representing the audio signal (amplitude over time).
    # sr: The sampling rate of the loaded audio file (set to 22050 Hz).
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    
    # mfccs: A 2D matrix where rows represent time and columns represent spectral features.
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    
    # padded: The final version of the MFCC matrix, normalized to MAX_LEN using zero-padding or truncation.
    if mfccs.shape[0] < MAX_LEN:
        padded = np.pad(mfccs, ((0, MAX_LEN - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        padded = mfccs[:MAX_LEN, :]
        
    return padded.reshape(1, MAX_LEN, N_MFCC)

# ==========================================
# --- WEB ROUTES (HTML PAGES) ---
# ==========================================

@app.route("/", methods=["GET", "POST"])
def login():
    """Handles user authentication and redirection based on role."""
    if request.method == "POST":
        # username: The string identifier provided by the user in the login form.
        username = request.form.get("username", "")
        # password: The secret string provided by the user in the login form.
        password = request.form.get("password", "")
        
        # user: A reference to the specific user's data dictionary found in the USERS database.
        user = USERS.get(username)
        if user and user["password"] == password:
            # session["username"]: Storing the user identity in the browser session for persistent access.
            session["username"] = username
            # session["role"]: Storing the user's role to control access to specific dashboards.
            session["role"] = user["role"]
            
            if session["role"] == "commander":
                return redirect(url_for("commander_dashboard"))
            
            return redirect(url_for("soldier_panel"))
            
        return "שם משתמש או סיסמה שגויים", 401
            
    return render_template("login.html")

@app.route("/soldier")
def soldier_panel():
    """Renders the soldier's broadcasting panel."""
    if "username" not in session or session["role"] != "soldier":
        return redirect(url_for("login"))
    
    return render_template("soldier.html", username=session["username"])

@app.route("/commander")
def commander_dashboard():
    """Renders the commander's live status dashboard."""
    if "username" not in session or session["role"] != "commander":
        return redirect(url_for("login"))
    
    return render_template("commander.html")

@app.route("/logout")
def logout():
    """Clears the user session and returns to login."""
    session.clear()
    return redirect(url_for("login"))


# ==========================================
# --- API ROUTES (DATA FLOW) ---
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():
    """Receives audio from a soldier, runs it through the model, and updates status."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # username: The name of the reporting soldier, retrieved from the form data.
    username = request.form.get("username", "unknown_soldier")
    # audio_bytes: The binary stream of the uploaded audio file.
    audio_bytes = request.files["file"].read()
    
    # input_data: The processed tensor (4D/3D array) formatted for neural network inference.
    input_data = preprocess_audio(audio_bytes)
    
    # predictions: An array of probability scores returned by the model's output layer.
    predictions = model.predict(input_data)
    
    # class_index: The index of the category that received the highest probability.
    class_index = int(np.argmax(predictions))
    # confidence: The decimal probability value of the most likely class.
    confidence = float(predictions[0][class_index])
    # detected_class: The string label corresponding to the identified sound class.
    detected_class = labels[class_index]

    # current_time: A string representing the exact time the detection was processed.
    current_time = time.strftime("%H:%M:%S")
    
    # Updating the global state with the new detection results.
    live_field_status[username] = {
        "status": detected_class,
        "confidence": round(confidence * 100, 2),
        "time": current_time
    }

    return jsonify({"status": "success", "detected": detected_class})

@app.route("/api/field_status", methods=["GET"])
def get_field_status():
    """Returns the live field status map for the commander's dashboard."""
    if "username" not in session or session["role"] != "commander":
        return jsonify({"error": "Unauthorized"}), 403
    
    return jsonify(live_field_status)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
