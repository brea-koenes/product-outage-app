import streamlit as st
import joblib

# Load the model
model = joblib.load("final_model.pkl")

st.title("Product Outage Classifier")

st.write("Enter a post, and the model will classify it.")

# Text input from the user
user_input = st.text_area("Enter your post here")

# Predict when button is clicked
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Wrap input in a list because most models expect iterable input
        prediction = model.predict([user_input])
        st.success(f"Predicted Label: {prediction[0]}")
