
import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load trained classifiers
political_model = joblib.load("political_embedding_model.pkl")
regional_model = joblib.load("regional_embedding_model.pkl")
socioeconomic_model = joblib.load("socioeconomic_embedding_model.pkl")

# App title
st.title("Media Framing Bias Detector")

st.write(
    "Enter a news headline to predict Political, Regional, "
    "and Socioeconomic framing."
)

# User input
headline = st.text_area("Enter news headline:")

# Prediction button
if st.button("Analyse"):

    if headline.strip():

        # Convert headline into embedding
        embedding = embedding_model.encode([headline])

        # Predict framing
        political_pred = political_model.predict(embedding)[0]
       regional_prob = regional_model.predict_proba(embedding)[0][1]

      if 0.40 <= regional_prob <= 0.60:
       regional_pred = "Uncertain"
     else:
      regional_pred = 1 if regional_prob > 0.60 else 0
        socioeconomic_pred = socioeconomic_model.predict(embedding)[0]

        st.subheader("Prediction Results")

        st.write(
            "Political framing:",
            "Detected" if political_pred == 1 else "Not detected"
        )

        st.write(
            "Regional framing:",
            "Detected" if regional_pred == 1 else "Not detected"
        )

        st.write(
            "Socioeconomic framing:",
            "Detected" if socioeconomic_pred == 1 else "Not detected"
        )

    else:
        st.warning("Please enter a headline.")
