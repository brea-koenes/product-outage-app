import streamlit as st
import joblib
import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Load the saved model and vectorizers
model = joblib.load("final_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
phraser = joblib.load("phraser.pkl")


# Set best threshold from training
BEST_THRESHOLD = 0.65  # use your actual best_thresh if different


# Define preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    phrased = phraser[tokens]
    return " ".join(phrased)


# Streamlit UI
st.title("Product Outage Classifier")
st.write("Enter a customer post to check if it's related to a product outage.")


user_input = st.text_area("Enter post here:")


if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        processed = preprocess_text(user_input)
        X = vectorizer.transform([processed])
        prob = model.predict_proba(X)[0][1]
        pred = int(prob >= BEST_THRESHOLD)


        label = "Outage (1)" if pred == 1 else "Not Outage (0)"
        st.success(f"Prediction: {label} | Probability: {prob:.2f}")



