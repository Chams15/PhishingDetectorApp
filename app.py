import streamlit as st
import joblib
import pandas as pd

# Load the saved model, vectorizer, and class names
@st.cache_resource
def load_components():
    model = joblib.load('mlp_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    class_names_loaded = joblib.load('class_names.pkl')
    return model, vectorizer, class_names_loaded

model, vectorizer, class_names = load_components()

st.set_page_config(page_title="Malicious URL Detector", page_icon=":lock:")

st.title("🕵️ Malicious URL Detector")
st.write("Enter a URL below to predict if it's benign, defacement, malware, or phishing.")

user_url = st.text_input("", placeholder="e.g., https://www.example.com/malicious-link")

if st.button("Analyze URL"):
    if user_url:
        with st.spinner('Analyzing URL...'):
            # Preprocess the input URL using the loaded vectorizer
            url_vectorized = vectorizer.transform([user_url])

            # Make prediction
            prediction_label = model.predict(url_vectorized)[0]
            prediction_proba = model.predict_proba(url_vectorized)[0]

            # Get the predicted class name
            predicted_class = class_names[prediction_label]

            st.success(f"The URL is predicted to be: **{predicted_class.capitalize()}**")

            # Display probabilities for each class
            st.subheader("Prediction Probabilities:")
            proba_df = pd.DataFrame({
                'Class': class_names,
                'Probability': prediction_proba
            }).sort_values(by='Probability', ascending=False)
            st.dataframe(proba_df.style.format({'Probability': '{:.2%}'}))
    else:
        st.warning("Please enter a URL to analyze.")

st.markdown("---")
st.info("This detector is for educational purposes and may not be 100 percent accurate.")