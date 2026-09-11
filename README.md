# Media Framing Bias Detector

This project detects three types of framing in news headlines:

- Political framing
- Regional framing
- Socioeconomic framing

The project compares classical NLP approaches such as TF-IDF with sentence embeddings.

Sentence embeddings performed better overall and were used for the deployed Streamlit application.

## Model

The application uses:

- SentenceTransformer: `all-MiniLM-L6-v2`
- Logistic Regression classifiers
- Separate models for Political, Regional, and Socioeconomic framing

## App

Enter a news headline and click **Analyse**.

The app returns whether each type of framing is:

- Detected
- Not detected

## Files

- `app.py` - Streamlit application
- `political_embedding_model.pkl`
- `regional_embedding_model.pkl`
- `socioeconomic_embedding_model.pkl`
- `requirements.txt`

## Limitations

The dataset is relatively small and model performance varies across different news events. Event-level distribution shift and class imbalance remain important limitations.
The deployed Streamlit prototype uses the sentence-embedding classifiers, while the fine-tuned DistilBERT models were evaluated separately in the modelling notebook.
Live Demo:
https://3x35k7rhsdvxjucwt7zg8t.streamlit.app/
