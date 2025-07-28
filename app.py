import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import streamlit as st

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load model and pre-processing tools
model = joblib.load("final_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
phraser = joblib.load("phraser.pkl")

# Text preprocessing function
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

# Streamlit app interface
st.title("Product Outage Classifier")
st.write("Enter a text post to classify whether it's related to a product outage.")

user_input = st.text_area("Text input:")

# Replace this with your best threshold found during training
BEST_THRESHOLD = 0.65

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        processed = preprocess_text(user_input)
        transformed = vectorizer.transform([processed])
        proba = model.predict_proba(transformed)[0][1]
        prediction = int(proba >= BEST_THRESHOLD)

        label = "Outage (1)" if prediction == 1 else "Not Outage (0)"
        st.success(f"Prediction: {label}  |  Probability: {proba:.2f}")


