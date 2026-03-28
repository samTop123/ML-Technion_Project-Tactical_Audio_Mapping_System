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

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Required for Sessions (Login)

# Audio processing constants
MAX_LEN = 431
N_MFCC = 40
INPUT_SHAPE = (MAX_LEN, N_MFCC)

# Mock Database for the Presentation
USERS = {
    "soldier1": {"password": "123", "role": "soldier"},
    "soldier2": {"password": "123", "role": "soldier"},
    "commander1": {"password": "admin", "role": "commander"}
}

# Global state for live tracking
# e.g., {"soldier1": {"status": "Gunfire", "confidence": 0.95, "time": "14:05:00"}}
live_field_status = {}


# ==========================================
# --- MODEL LOADING ---
# ==========================================

print("Loading model...")
model = tf.keras.models.load_model("model_train_clean_cnn.keras")
print("Model loaded successfully.")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]


# ==========================================
# --- HELPER FUNCTIONS ---
# ==========================================

def preprocess_audio(audio_bytes):
    """Converts raw audio bytes into padded MFCC features for the model."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    
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
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        
        # Validate credentials
        user = USERS.get(username)
        if user and user["password"] == password:
            session["username"] = username
            session["role"] = user["role"]
            
            # Redirect to appropriate dashboard
            if session["role"] == "commander":
                return redirect(url_for("commander_dashboard"))
            
            return redirect(url_for("soldier_panel"))
            
        return "שם משתמש או סיסמה שגויים", 401
            
    # If GET request, show login form
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
    print(">>> 1. Sound request received from soldier! <<<")
    
    if "file" not in request.files:
        print(">>> Error: No file received <<<")
        return jsonify({"error": "No file provided"}), 400

    username = request.form.get("username", "unknown_soldier")
    audio_bytes = request.files["file"].read()
    print(">>> 2. File read successfully from request <<<")
    
    # Processing stage
    print(">>> 3. Starting audio preprocessing... <<<")
    input_data = preprocess_audio(audio_bytes)
    print(">>> 4. Audio preprocessing finished! Starting model prediction... <<<")
    
    # Model stage
    predictions = model.predict(input_data)
    print(">>> 5. Model finished prediction successfully! <<<")
    
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[0][class_index])
    detected_class = labels[class_index]

    # Update the global status map for the commander
    current_time = time.strftime("%H:%M:%S")
    live_field_status[username] = {
        "status": detected_class,
        "confidence": round(confidence * 100, 2),
        "time": current_time
    }

    print(">>> 6. Updating commander and returning response to browser <<<")
    return jsonify({"status": "success", "detected": detected_class})

@app.route("/api/field_status", methods=["GET"])
def get_field_status():
    """Returns the live field status map for the commander's dashboard."""
    if "username" not in session or session["role"] != "commander":
        return jsonify({"error": "Unauthorized"}), 403
    
    return jsonify(live_field_status)


# ==========================================
# --- APP EXECUTION ---
# ==========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)