import joblib
import numpy as np
from src.extract_features import extract_nlp_features, extract_tabular_features

# Load trained model
model = joblib.load("models/xgb_model.pkl")

# New pitch input (you can change this!)
new_pitch = "A decentralized AI protocol for clinical trials"
new_meta = {
    "team_size": 5,
    "twitter_followers": 2500,
    "past_exits": 2,
    "has_prototype": 1
}

# Extract features
nlp_features = extract_nlp_features(new_pitch)
tabular_features = extract_tabular_features(new_meta)
X = np.concatenate((nlp_features, tabular_features)).reshape(1, -1)

# Predict
prob = model.predict_proba(X)[0][1]  # Probability of class "1" (success)
print(f"🚀 Fundraise success probability: {prob:.2%}")
