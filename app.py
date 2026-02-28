import streamlit as st
import joblib
import numpy as np
import re
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="AI Fake News Detector", layout="wide")

# Load Models
lr_model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
dl_model = load_model("dl_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

st.title("🧠 Hybrid AI-Based Fake News Detection System")

col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("Enter News Article Text")

    if st.button("Analyze"):
        cleaned = clean_text(input_text)

        # Logistic Regression Prediction
        tfidf_input = vectorizer.transform([cleaned])
        lr_pred = lr_model.predict(tfidf_input)[0]

        # Deep Learning Prediction
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=300)
        dl_pred = dl_model.predict(padded)[0][0]

        confidence = float(dl_pred if dl_pred > 0.5 else 1 - dl_pred)

        if dl_pred > 0.5:
            st.success("Deep Learning Prediction: REAL News")
        else:
            st.error("Deep Learning Prediction: FAKE News")

        st.write(f"Confidence Score: {confidence*100:.2f}%")

        st.write("Baseline ML Prediction:",
                 "REAL" if lr_pred == 1 else "FAKE")

with col2:
    st.subheader("Prediction Probability Chart")

    if 'dl_pred' in locals():
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], [1-dl_pred, dl_pred])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

st.markdown("---")
st.subheader("Model Comparison")
st.write("Logistic Regression (Baseline ML)")
st.write("LSTM Deep Learning Model")