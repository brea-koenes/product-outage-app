import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np # Import numpy

# Download necessary NLTK data
# These should ideally be pre-downloaded or handled in a Streamlit deployment setup
# For Colab environment for testing, we can include them here, but be mindful for deployment
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')


# Define stopwords and lemmatizer (needs to match training)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define preprocessing function (needs to match training)
def preprocess_text_for_phrasing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

# Load the pickled model and TF-IDF components
# Ensure these paths are correct relative to where your Streamlit app will run,
# or update them to point to your Google Drive if accessed during deployment setup.
# For a typical Streamlit deployment on platforms like Hugging Face Spaces or Render,
# you'll likely need to upload these .pkl files along with your app.py.
try:
    model_path = 'final_model.pkl' # Adjust path as needed for deployment
    tfidf_vectorizer_path = 'tfidf_vectorizer.pkl' # Adjust path as needed for deployment
    phraser_path = 'phraser.pkl' # Adjust path as needed for deployment

    final_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    phraser = joblib.load(phraser_path)
    st.success("Model and components loaded successfully!")
except FileNotFoundError:
    st.error("Error loading model or TF-IDF components. Make sure the .pkl files are in the correct directory.")
    st.stop() # Stop the app if files are not found
except Exception as e:
    st.error(f"An error occurred during loading: {e}")
    st.stop()


# Define the best threshold found during training
# This should be saved and loaded along with the model for consistency
# For now, using the value from the notebook. You might want to save this value.
# Assuming best_thresh was determined in your notebook
best_thresh = 0.5 # Replace with your actual best_thresh value or load it


# Streamlit app interface
st.title("Starbucks Product Outage Classifier")
st.write("Enter a customer post to classify if it indicates a product outage.")

# Text input area
user_input = st.text_area("Enter post text here:")

if st.button("Classify"):
    if user_input:
        # Preprocess the input text
        input_tokens = preprocess_text_for_phrasing(user_input)
        input_phrased = phraser[input_tokens]
        input_tfidf = ' '.join(input_phrased)

        # Transform the preprocessed text using the fitted TF-IDF vectorizer
        # Need to handle cases where input text results in no features found in the vocabulary
        try:
            input_tfidf_vector = tfidf_vectorizer.transform([input_tfidf]).toarray()

            # Make a prediction using the final model
            prediction_proba = final_model.predict_proba(input_tfidf_vector)[:, 1]
            prediction = (prediction_proba >= best_thresh).astype(int)[0]

            # Display the result
            st.subheader("Classification Result:")
            if prediction == 1:
                st.error(f"This post likely indicates a **Product Outage** (Probability: {prediction_proba[0]:.4f})")
            else:
                st.success(f"This post likely **Does Not** indicate a product outage (Probability: {prediction_proba[0]:.4f})")

            st.write(f"Raw Probability of Class 1: {prediction_proba[0]:.4f}")

        except ValueError as ve:
            st.warning(f"Could not classify the post. It may not contain words found in the training vocabulary. Error: {ve}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    else:
        st.warning("Please enter some text to classify.")

# Optional: Add information about the model or how it works
st.sidebar.header("About")
st.sidebar.info("This app uses an XGBoost model trained on customer posts to classify potential Starbucks product outages.")



