import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and transformers
model = joblib.load("final_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
phraser = joblib.load("phraser.pkl")

# Define text preprocessing (match your training pipeline)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    phrased_tokens = phraser[tokens]
    return " ".join(phrased_tokens)

# Streamlit UI
st.title("Product Outage Classifier")
st.write("Enter a customer post to classify it as an outage (1) or not (0).")

user_input = st.text_area("Enter post text here:")

BEST_THRESHOLD = 0.65  # replace with your best_thresh

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter text.")
    else:
        processed = preprocess_text(user_input)
        tfidf = vectorizer.transform([processed])
        proba = model.predict_proba(tfidf)[0][1]
        label = int(proba >= BEST_THRESHOLD)
        st.success(f"Prediction: {label} (Probability: {proba:.2f})")

