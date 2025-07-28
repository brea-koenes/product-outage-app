import streamlit as st
import joblib


# Load the model and vectorizer
model = joblib.load("final_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Best threshold found during training
BEST_THRESHOLD = 0.65  # 🔁 Replace this with your actual best_thresh


st.title("Product Outage Post Classifier")
st.write("Enter a text post and the model will classify it as an outage (1) or not (0).")


# User input
user_input = st.text_area("Enter your post:")


if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Vectorize input text
        X_input = vectorizer.transform([user_input])


        # Predict probability
        prob = model.predict_proba(X_input)[0][1]  # Probability of class 1


        # Apply threshold
        prediction = int(prob >= BEST_THRESHOLD)


        st.success(f"Prediction: {prediction} (probability = {prob:.2f})")

        st.success(f"Prediction: {label} | Probability: {prob:.2f}")



