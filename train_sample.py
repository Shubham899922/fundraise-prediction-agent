import os
import sys
import numpy as np
import joblib

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from extract_features import extract_nlp_features, extract_tabular_features
from train_model import train_model

print("🔁 Starting training script...")

# Example input data
pitches = ["Decentralized identity system", "A social app for DeFi investors"]
meta = [
    {"team_size": 3, "twitter_followers": 1000, "past_exits": 1, "has_prototype": 1},
    {"team_size": 2, "twitter_followers": 300, "past_exits": 0, "has_prototype": 0}
]
labels = [1, 0]

print("📦 Data loaded...")

# Convert inputs to features
X = []
for i in range(len(pitches)):
    print(f"🔍 Processing pitch #{i+1}")
    nlp = extract_nlp_features(pitches[i])
    tab = extract_tabular_features(meta[i])
    X.append(np.concatenate((nlp, tab)))

print("✅ Features extracted.")

# Train model
model = train_model(np.array(X), np.array(labels))
print("📈 Model trained.")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")

print("✅ Model trained and saved!")
