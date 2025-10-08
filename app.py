from flask import Flask, request, jsonify, render_template
import joblib
import librosa
import numpy as np
import os

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__, template_folder='templates')

# -----------------------------
# Load Model & Scaler
# -----------------------------
model_filename = 'speech_emotion_model.joblib'
scaler_filename = 'scaler.joblib'

try:
    loaded_model = joblib.load(model_filename)
    print(f"✅ Model successfully loaded from {model_filename}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    loaded_model = None

try:
    scaler = joblib.load(scaler_filename)
    print(f"✅ Scaler successfully loaded from {scaler_filename}")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler = None


# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(y, sr):
    """
    Extracts 10 MFCC features — must match training setup.
    """
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10).T, axis=0)
    return mfcc


# -----------------------------
# Prediction Function
# -----------------------------
def predict_emotion(audio_file_path, model, scaler, sampling_rate=22050):
    if model is None:
        print("❌ Model not loaded.")
        return None

    try:
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=sampling_rate)
        y = librosa.util.normalize(y)

        # Extract features
        features = extract_features(y, sr).reshape(1, -1)

        # Scale features
        if scaler is not None:
            features = scaler.transform(features)

        # Predict emotion
        prediction = model.predict(features)
        return prediction[0]

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save temporarily
    file_path = os.path.join('temp_audio.wav')
    audio_file.save(file_path)

    # Predict emotion
    predicted_emotion = predict_emotion(file_path, loaded_model, scaler)

    # Clean up
    os.remove(file_path)

    if predicted_emotion:
        return jsonify({'emotion': predicted_emotion})
    else:
        return jsonify({'error': 'Could not predict emotion'}), 500


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
