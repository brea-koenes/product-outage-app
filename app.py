import streamlit as st
import joblib
import numpy as np


# Load model
model = joblib.load("final_model.pkl")


st.title("Product Outage Classifier")


st.write("Enter product features to predict an outage.")


# Example inputs – update these based on your model's features
feature_1 = st.number_input("Feature 1")
feature_2 = st.number_input("Feature 2")
feature_3 = st.number_input("Feature 3")


if st.button("Predict"):
    input_data = np.array([[feature_1, feature_2, feature_3]])
    prediction = model.predict(input_data)


    st.success(f"Prediction: {prediction[0]}")


