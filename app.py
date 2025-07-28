import streamlit as st
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb


# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')


# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Define preprocessing function (same as used in training)
def preprocess_text_for_phrasing(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = word_tokenize(text)  # tokenize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]  # lemmatize and remove stop words
    return tokens


# Load the pickled artifacts
@st.cache_resource
def load_artifacts():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('phraser.pkl', 'rb') as f:
        phraser = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('best_threshold.pkl', 'rb') as f:
        best_thresh = pickle.load(f)
    return vectorizer, phraser, model, best_thresh


# Load artifacts
tfidf_vectorizer, phraser, xgb_clf, best_thresh = load_artifacts()


# Streamlit app layout
st.title("Text Classification App")
st.write("Enter text below to classify it as positive (1) or negative (0).")


# Text input
user_input = st.text_area("Input Text", height=200)


# Predict button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Preprocess the input text
        tokens = preprocess_text_for_phrasing(user_input)
        # Apply phraser
        phrased_tokens = phraser[tokens]
        # Rejoin tokens for TF-IDF
        text_tfidf = ' '.join(phrased_tokens)
        # Transform using TF-IDF vectorizer
        text_transformed = tfidf_vectorizer.transform([text_tfidf]).toarray()
        
        # Predict probability
        proba = xgb_clf.predict_proba(text_transformed)[:, 1]
        
        # Apply best threshold
        prediction = (proba >= best_thresh).astype(int)[0]
        
        # Display results
        st.write(f"**Prediction**: {'Positive (1)' if prediction == 1 else 'Negative (0)'}")
        st.write(f"**Probability of Positive Class**: {proba[0]:.4f}")



