import sys, os
# Ensure Python can import from src/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import pandas as pd
import joblib

from extract_features import extract_nlp_features, extract_tabular_features


# Load trained model
model = joblib.load("models/xgb_model.pkl")

def explain_prediction(prob, pitch, meta):
    reasons = []
    if prob > 0.75:
        reasons.append("✅ Strong success signal overall.")
    elif prob < 0.35:
        reasons.append("⚠️ Low success probability, possibly due to weak signals.")
    if len(pitch.split()) > 10:
        reasons.append("🧠 Detailed pitch with sufficient narrative.")
    else:
        reasons.append("💡 Pitch may be too short or unclear.")
    if meta["twitter_followers"] > 1000:
        reasons.append("📣 Good social traction.")
    if meta["past_exits"] > 0:
        reasons.append("🧑‍💼 Experienced founder(s) with exits.")
    if meta["has_prototype"]:
        reasons.append("🛠 Prototype ready — reduces risk.")
    return " | ".join(reasons)

st.set_page_config(page_title="Fundraise Prediction Agent", layout="centered")
st.title("🚀 Fundraise Prediction Agent")

# — Single Project Prediction —
st.header("🔍 Predict One Project")
pitch = st.text_input("Enter your project pitch")
team_size = st.number_input("Team size", min_value=1)
twitter_followers = st.number_input("Twitter followers", min_value=0)
past_exits = st.number_input("Past startup exits", min_value=0)
has_prototype = st.checkbox("Do you have a working prototype?")

if st.button("🔮 Predict Fundraise Success"):
    meta = {
        "team_size": team_size,
        "twitter_followers": twitter_followers,
        "past_exits": past_exits,
        "has_prototype": int(has_prototype)
    }
    nlp = extract_nlp_features(pitch)
    tab = extract_tabular_features(meta)
    X = np.concatenate((nlp, tab)).reshape(1, -1)

    prob = model.predict_proba(X)[0][1]
    st.success(f"📊 Success Probability: **{prob:.2%}**")

    explanation = explain_prediction(prob, pitch, meta)
    st.info(f"📌 Why this score? {explanation}")


# — Batch CSV Prediction —
st.markdown("---")
st.header("📁 Predict from CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    results = []
    explanations = []

    for _, row in df.iterrows():
        pitch = row["pitch"]
        meta = {
            "team_size": row["team_size"],
            "twitter_followers": row["twitter_followers"],
            "past_exits": row["past_exits"],
            "has_prototype": row["has_prototype"]
        }
        nlp = extract_nlp_features(pitch)
        tab = extract_tabular_features(meta)
        X = np.concatenate((nlp, tab)).reshape(1, -1)

        prob = model.predict_proba(X)[0][1]
        results.append(prob)
        explanations.append(explain_prediction(prob, pitch, meta))

    df["success_probability"] = [f"{p:.2%}" for p in results]
    df["explanation"] = explanations

    st.write("📊 Prediction Results")
    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv_data,
        file_name="predictions.csv",
        mime="text/csv"
    )
