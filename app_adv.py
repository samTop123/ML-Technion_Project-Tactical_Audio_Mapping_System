from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import tensorflow as tf
import numpy as np
import librosa
import io
import time

app = Flask(__name__)
app.secret_key = "super_secret_key_for_project" # חובה בשביל Sessions (Login)

# --- Mock Database for the Presentation ---
# In a real app, this would be SQLite or MongoDB
USERS = {
    "soldier1": {"password": "123", "role": "soldier"},
    "soldier2": {"password": "123", "role": "soldier"},
    "commander1": {"password": "admin", "role": "commander"}
}

# This will store the latest prediction for each soldier so the commander can see it
# e.g., {"soldier1": {"status": "Gunfire", "confidence": 0.95, "time": "14:05:00"}}
live_field_status = {}

# --- 1 & 2 & 3. Load Model and Labels (Keep your existing code) ---
input_shape = (431, 40)
model = tf.keras.models.load_model("model_train_clean_cnn.keras")
print("Model loaded successfully.")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- 4. Preprocess (Keep your existing code) ---
MAX_LEN = 431
N_MFCC = 40
def preprocess_audio(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    if mfccs.shape[0] < MAX_LEN:
        padded = np.pad(mfccs, ((0, MAX_LEN - mfccs.shape[0]), (0,0)), mode='constant')
    else:
        padded = mfccs[:MAX_LEN, :]
    return padded.reshape(1, MAX_LEN, N_MFCC)

# ==========================================
# --- WEB ROUTES (HTML PAGES) ---
# ==========================================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        # Check credentials
        if username in USERS and USERS[username]["password"] == password:
            session["username"] = username
            session["role"] = USERS[username]["role"]
            
            # Redirect based on role
            if session["role"] == "commander":
                return redirect(url_for("commander_dashboard"))
            else:
                return redirect(url_for("soldier_panel"))
        else:
            return "שם משתמש או סיסמה שגויים", 401
            
    # If GET, show the login page
    return render_template("login.html")

@app.route("/soldier")
def soldier_panel():
    if "username" not in session or session["role"] != "soldier":
        return redirect(url_for("login"))
    return render_template("soldier.html", username=session["username"])

@app.route("/commander")
def commander_dashboard():
    if "username" not in session or session["role"] != "commander":
        return redirect(url_for("login"))
    return render_template("commander.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==========================================
# --- API ROUTES (DATA FLOW) ---
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():
    print(">>> 1. הגיעה בקשת סאונד מהחייל! <<<")
    
    if "file" not in request.files:
        print(">>> שגיאה: לא התקבל קובץ <<<")
        return jsonify({"error": "No file provided"}), 400

    username = request.form.get("username", "unknown_soldier")
    audio_bytes = request.files["file"].read()
    print(">>> 2. הקובץ נקרא בהצלחה מתוך הבקשה <<<")
    
    # שלב העיבוד - כאן בדרך כלל יש בעיות עם קבצי סאונד
    print(">>> 3. מתחיל לעבד את הסאונד (preprocess)... <<<")
    input_data = preprocess_audio(audio_bytes)
    print(">>> 4. סיימתי לעבד את הסאונד! מתחיל חיזוי מודל... <<<")
    
    # שלב המודל
    predictions = model.predict(input_data)
    print(">>> 5. המודל סיים לחזות בהצלחה! <<<")
    
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

    print(">>> 6. מעדכן את המפקד ומחזיר תשובה לדפדפן <<<")
    return jsonify({"status": "success", "detected": detected_class})

@app.route("/api/field_status", methods=["GET"])
def get_field_status():
    # The Commander's dashboard will poll this URL to get live updates
    if "username" not in session or session["role"] != "commander":
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(live_field_status)

# --- 6. Run server ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
