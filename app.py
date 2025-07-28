{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 ArialMT;\f1\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs29\fsmilli14667 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import streamlit as st
\f1\fs24 \

\f0\fs29\fsmilli14667 import joblib
\f1\fs24 \

\f0\fs29\fsmilli14667 import numpy as np
\f1\fs24 \
\

\f0\fs29\fsmilli14667 # Load model
\f1\fs24 \

\f0\fs29\fsmilli14667 model = joblib.load("final_model.pkl")
\f1\fs24 \
\

\f0\fs29\fsmilli14667 st.title("Product Outage Classifier")
\f1\fs24 \
\

\f0\fs29\fsmilli14667 st.write("Enter product features to predict an outage.")
\f1\fs24 \
\

\f0\fs29\fsmilli14667 # Example inputs \'96 update these based on your model's features
\f1\fs24 \

\f0\fs29\fsmilli14667 feature_1 = st.number_input("Feature 1")
\f1\fs24 \

\f0\fs29\fsmilli14667 feature_2 = st.number_input("Feature 2")
\f1\fs24 \

\f0\fs29\fsmilli14667 feature_3 = st.number_input("Feature 3")
\f1\fs24 \
\

\f0\fs29\fsmilli14667 if st.button("Predict"):
\f1\fs24 \

\f0\fs29\fsmilli14667 \'a0\'a0\'a0\'a0input_data = np.array([[feature_1, feature_2, feature_3]])
\f1\fs24 \

\f0\fs29\fsmilli14667 \'a0\'a0\'a0\'a0prediction = model.predict(input_data)
\f1\fs24 \
\

\f0\fs29\fsmilli14667 \'a0\'a0\'a0\'a0st.success(f"Prediction: \{prediction[0]\}")
\f1\fs24 \
}