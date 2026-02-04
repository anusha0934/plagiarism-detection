import streamlit as st
import nltk
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (needed for Streamlit Cloud)
nltk.download('stopwords')
nltk.download('wordnet')
# Load model & vectorizer
model = joblib.load("models/plagiarism_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

st.title("📝 AI vs Human Text Detector")

user_text = st.text_area("Enter text here:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max() * 100

        if pred == 1:
            st.error(f"🤖 AI Generated ({prob:.2f}% confidence)")
        else:
            st.success(f"✍️ Human Written ({prob:.2f}% confidence)")
