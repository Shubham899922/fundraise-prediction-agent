import sys, os
# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import pandas as pd
from extract_features import extract_nlp_features, extract_tabular_features

st.set_page_config(page_title="Fundraise Prediction Agent", layout="centered")
st.title("🚀 Fundraise Prediction Agent")

def explain_prediction(prob, pitch, meta):
    reasons = []
    if prob > 0.75:
        reasons.append("✅ Strong success signal.")
    elif prob < 0.35:
        reasons.append("⚠️ Low probability.")
    reasons.append("🧠 Detailed pitch." if len(pitch.split()) > 10 else "💡 Pitch is short.")
    if meta["twitter_followers"] > 1000:
        reasons.append("📣 Good traction.")
    if meta["past_exits"] > 0:
        reasons.append("🧑‍💼 Experienced founders.")
    if meta["has_prototype"]:
        reasons.append("🛠 Prototype readiness.")
    return " | ".join(reasons)

st.header("🔍 Predict Project Fundraise Success")
pitch = st.text_input("Enter project pitch")
team_size = st.number_input("Team size", min_value=1, value=1)
twitter_followers = st.number_input("Twitter followers", min_value=0, value=0)
past_exits = st.number_input("Past startup exits", min_value=0, value=0)
has_prototype = st.checkbox("Do you have a prototype?")

if st.button("🔮 Predict Success"):
    # Lazy-load heavy dependencies only when needed
    import joblib
    model = joblib.load("models/xgb_model.pkl")

    meta = {
        "team_size": team_size,
        "twitter_followers": twitter_followers,
        "past_exits": past_exits,
        "has_prototype": int(has_prototype),
    }

    nlp_vec = extract_nlp_features(pitch)
    tab_vec = extract_tabular_features(meta)
    X = np.concatenate((nlp_vec, tab_vec)).reshape(1, -1)

    prob = model.predict_proba(X)[0][1]
    st.success(f"📊 Success Probability: **{prob:.2%}**")
    st.info(explain_prediction(prob, pitch, meta))
