import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

political_model = joblib.load("political_embedding_model.pkl")
regional_model = joblib.load("regional_embedding_model.pkl")
socioeconomic_model = joblib.load("socioeconomic_embedding_model.pkl")
thresholds = joblib.load("thresholds.pkl")

st.title("Media Framing Bias Detector")

st.write(
    "Enter a news headline to predict Political, Regional, "
    "and Socioeconomic framing."
)

headline = st.text_area("Enter news headline:")

if st.button("Analyse"):

    if headline.strip():

        embedding = embedding_model.encode([headline])

        political_prob = political_model.predict_proba(embedding)[0][1]
        regional_prob = regional_model.predict_proba(embedding)[0][1]
        socioeconomic_prob = socioeconomic_model.predict_proba(embedding)[0][1]

        political_pred = 1 if political_prob >= thresholds["political"] else 0

        if 0.40 <= regional_prob <= 0.60:
            regional_pred = "Uncertain"
        else:
            regional_pred = 1 if regional_prob > 0.60 else 0

        socioeconomic_pred = 1 if socioeconomic_prob >= thresholds["socioeconomic"] else 0

        st.subheader("Prediction Results")

        st.write(
            "Political framing:",
            "Detected" if political_pred == 1 else "Not detected"
        )

        st.write(
            "Regional framing:",
            "Uncertain" if regional_pred == "Uncertain" else ("Detected" if regional_pred == 1 else "Not detected")
        )

        st.write(
            "Socioeconomic framing:",
            "Detected" if socioeconomic_pred == 1 else "Not detected"
        )

    else:
        st.warning("Please enter a headline.")
