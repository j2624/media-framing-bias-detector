# ==========================================================================
# RETRAIN + FIX: Media Framing Bias Detector
# ==========================================================================
# Run this as ONE cell in Colab, top to bottom. It will:
#   1) Load your labelled dataset
#   2) Rebuild sentence embeddings (same model your app uses)
#   3) Retrain all 3 models -- Regional now WITHOUT class_weight="balanced"
#      (Political and Socioeconomic keep "balanced", since their labels
#      were clean -- 85% and 90% blind-relabel consistency. Regional was
#      only 25% consistent, so "balanced" weighting was pushing it to
#      predict "Regional framing detected" on almost every headline.)
#   4) Save the 3 .pkl files + a fresh app.py
#   5) Run 5 test headlines through the fixed models so you can SEE the
#      difference immediately, before even opening Streamlit.
#
# BEFORE RUNNING: upload `recovered_manual_labelled_235_transformer.csv`
# to your Colab session (left sidebar -> Files -> upload), so it sits at
# /content/recovered_manual_labelled_235_transformer.csv
# ==========================================================================

import re
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report

DATA_PATH = "/content/recovered_manual_labelled_235_transformer.csv"

# --------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

label_cols = ["political_label", "regional_label", "socioeconomic_label"]
df[label_cols] = df[label_cols].astype(int)

# Same preprocessing used earlier in the notebook
df["processed_title"] = (
    df["title"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^\w\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("Loaded rows:", len(df))
for c in label_cols:
    print(c, "positive rate:", round(df[c].mean(), 3))

# --------------------------------------------------------------------
# 2. Embeddings (same model the Streamlit app uses at inference time)
# --------------------------------------------------------------------
from sentence_transformers import SentenceTransformer

print("\nLoading sentence-transformers/all-MiniLM-L6-v2 ...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

X_embeddings = embedding_model.encode(df["processed_title"].tolist())
groups = df["event"]

y_political = df["political_label"]
y_regional = df["regional_label"]
y_socioeconomic = df["socioeconomic_label"]

# --------------------------------------------------------------------
# 3. Sanity check: compare balanced vs. unweighted for Regional
#    using GroupKFold (same evaluation style as the rest of the notebook)
# --------------------------------------------------------------------
gkf = GroupKFold(n_splits=3)

def evaluate(class_weight, y, X=X_embeddings, groups=groups):
    all_true, all_pred = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
        model.fit(X[train_idx], y.iloc[train_idx])
        preds = model.predict(X[test_idx])
        all_true.extend(y.iloc[test_idx])
        all_pred.extend(preds)
    macro_f1 = f1_score(all_true, all_pred, average="macro")
    positive_rate = np.mean(all_pred)
    return macro_f1, positive_rate, all_true, all_pred

print("\n--- Regional: balanced vs. unweighted (out-of-fold) ---")
f1_bal, posrate_bal, yt, yp_bal = evaluate("balanced", y_regional)
f1_none, posrate_none, _, yp_none = evaluate(None, y_regional)

print(f"class_weight='balanced' -> Macro F1: {f1_bal:.3f} | predicted positive rate: {posrate_bal:.3f}")
print(f"class_weight=None       -> Macro F1: {f1_none:.3f} | predicted positive rate: {posrate_none:.3f}")
print(f"(actual positive rate in data: {y_regional.mean():.3f})")

print("\nclassification_report (class_weight=None):")
print(classification_report(yt, yp_none, zero_division=0))

# --------------------------------------------------------------------
# 4. Train FINAL models on all labelled data and save
# --------------------------------------------------------------------
targets = {
    "political": y_political,
    "regional": y_regional,
    "socioeconomic": y_socioeconomic,
}

class_weight_by_target = {
    "political": "balanced",
    "regional": None,          # <-- the fix
    "socioeconomic": "balanced",
}

final_models = {}

for target_name, y_target in targets.items():
    final_model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight_by_target[target_name],
        random_state=42,
    )
    final_model.fit(X_embeddings, y_target)
    final_models[target_name] = final_model
    joblib.dump(final_model, f"/content/{target_name}_embedding_model.pkl")
    print(f"Saved {target_name}_embedding_model.pkl (class_weight={class_weight_by_target[target_name]})")

# --------------------------------------------------------------------
# 5. Rewrite app.py (unchanged logic -- just re-saved for completeness)
# --------------------------------------------------------------------
app_code = '''
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
        regional_pred = regional_model.predict(embedding)[0]
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
'''

with open("/content/app.py", "w") as f:
    f.write(app_code)

print("\napp.py written to /content/app.py")

# --------------------------------------------------------------------
# 6. Live test -- prove the fix works before even opening Streamlit
# --------------------------------------------------------------------
test_headlines = [
    "Punjab farmers protest over new state agricultural policy",          # expect Regional-leaning
    "New smartphone launched with better battery life",                   # expect nothing detected
    "Prime Minister meets foreign leaders for bilateral talks",           # expect Political
    "Trade dispute raises concerns over jobs and household costs",        # expect Socioeconomic
    "Delhi records highest rainfall in a decade as monsoon intensifies",  # expect Regional
]

print("\n===== LIVE TEST PREDICTIONS =====")
test_embeddings = embedding_model.encode(test_headlines)

for text, emb in zip(test_headlines, test_embeddings):
    emb = emb.reshape(1, -1)
    pol = final_models["political"].predict(emb)[0]
    reg = final_models["regional"].predict(emb)[0]
    soc = final_models["socioeconomic"].predict(emb)[0]
    print(f"\nHeadline: {text}")
    print(f"  Political:     {'Detected' if pol == 1 else 'Not detected'}")
    print(f"  Regional:      {'Detected' if reg == 1 else 'Not detected'}")
    print(f"  Socioeconomic: {'Detected' if soc == 1 else 'Not detected'}")

print("\nDone. political_embedding_model.pkl, regional_embedding_model.pkl,")
print("socioeconomic_embedding_model.pkl and app.py are all in /content.")
print("Now just run:  !streamlit run app.py --server.port 8501")
