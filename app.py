import streamlit as st

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



