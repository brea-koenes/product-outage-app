import streamlit as st
import pickle
import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Load pickled objects
try:
    with open('lemmatizer.pkl', 'rb') as f:
        lemmatizer = pickle.load(f)
    with open('phraser.pkl', 'rb') as f:
        phraser = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('finalized_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    loaded_successfully = True
except FileNotFoundError as e:
    st.error(f"Error loading pickled file: {e}. Make sure the files are in the same directory.")
    lemmatizer, phraser, tfidf_vectorizer, xgb_model = None, None, None, None
    loaded_successfully = False

stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    if lemmatizer is None or phraser is None or tfidf_vectorizer is None:
        return None

    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    phrased_tokens = phraser[tokens]
    text_tfidf = ' '.join(phrased_tokens)
    tfidf_transformed = tfidf_vectorizer.transform([text_tfidf]).toarray()
    return tfidf_transformed

# Streamlit app layout
st.title('Text Classification App')

user_input = st.text_area("Enter text for classification:")

if st.button('Predict'):
    if user_input:
        if loaded_successfully:
            processed_text = preprocess_text(user_input)

            if processed_text is not None and xgb_model is not None:
                prediction = xgb_model.predict(processed_text)[0]
                if prediction == 1:
                    st.error("Predicted Label: 1 (Negative)")
                else:
                    st.success("Predicted Label: 0 (Positive)")
            elif processed_text is None:
                 st.error("Preprocessing failed. Please check the loaded objects.")
            else: # xgb_model is None
                 st.error("Model not loaded.")
        else:
            st.error("Required files failed to load. Please check the file paths and try again.")
    else:
        st.warning("Please enter some text to get a prediction.")

