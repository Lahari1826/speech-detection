# train_model.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Example / Dummy Training Data
# Replace this with your real features and labels if you have
# -----------------------------
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.choice(['happy', 'sad', 'angry'], size=100)  # 3 classes

# Optional: scale features (if you scale in Flask, match this)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train the model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# -----------------------------
# Save the model
# -----------------------------
joblib.dump(model, 'speech_emotion_model.joblib')
joblib.dump(scaler, 'scaler.joblib')  # Save scaler if you want to use it in Flask

print("✅ Model and scaler saved successfully!")
