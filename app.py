import streamlit as st
import joblib
import numpy as np
import os


st.set_page_config(
    page_title="Malicious URL Detector",
    page_icon="🛡️",
    layout="centered"
)


@st.cache_resource
def load_resources():
    try:
        
        model = joblib.load('mlp_model.pkl')
        
        
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        
        class_names = joblib.load('class_names.pkl')
        
        return model, vectorizer, class_names
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.error("Please ensure 'mlp_model.pkl', 'tfidf_vectorizer.pkl', and 'class_names.pkl' are in the same directory.")
        return None, None, None

model, vectorizer, class_names = load_resources()


st.title("🛡️ Malicious URL Detector")
st.markdown("""
This tool uses a **Multi-Layer Perceptron (Neural Network)** to analyze URLs and detect potential security threats such as phishing or malware.
""")

st.divider()

url_input = st.text_input("Enter a URL to scan:", placeholder="e.g., http://example.com")

if st.button("Analyze URL", type="primary"):
    if not url_input:
        st.warning("Please enter a URL first.")
    elif model is None or vectorizer is None:
        st.error("System resources could not be loaded. Please check the model files.")
    else:
        with st.spinner("Analyzing..."):
            try:
                
                processed_input = vectorizer.transform([url_input])
                
               
                prediction_index = model.predict(processed_input)[0]
                
                if isinstance(class_names, dict):
                    result_label = class_names.get(prediction_index, "Unknown")
                else:
                    result_label = class_names[prediction_index]

               
                st.subheader("Analysis Result:")
                
                
                safe_keywords = ['benign', 'safe', 'good']
                is_safe = any(keyword in str(result_label).lower() for keyword in safe_keywords)

                if is_safe:
                    st.success(f"✅ **Safe**: The URL is classified as: **{result_label}**")
                else:
                    st.error(f"⚠️ **Malicious**: The URL is classified as: **{result_label}**")
                    st.markdown("**Recommendation:** Do not click on this link.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

# --- Sidebar Info ---
with st.sidebar:
    st.header("About Model")
    st.info(f"**Model Type:** Multi-Layer Perceptron (MLP)")
    st.info("**Vectorization:** TF-IDF")
    st.markdown("---")
    st.caption("Ensure your .pkl files match the versions used during training.")