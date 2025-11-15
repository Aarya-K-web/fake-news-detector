import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (required for Streamlit Cloud)
nltk.download("stopwords")

ps = PorterStemmer()
stopwords_set = set(stopwords.words("english"))

# Load model and vectorizer
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()

# Preprocess function
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stopwords_set]
    return " ".join(words)

# ---------------- UI DESIGN ---------------- #
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Title Section
st.markdown(
    """
    <h1 style='text-align: center; color: white;'>📰 Fake News Detector</h1>
    <p style='text-align: center; font-size:18px;'>
        Enter a news headline below to check whether it is <b style="color:lightgreen;">Real</b> or 
        <b style="color:#ff4b4b;">Fake</b>.
    </p>
    """,
    unsafe_allow_html=True
)

# Input Box
headline = st.text_area("Enter headline:", "")

# Predict Button
if st.button("🔍 Predict"):
    if headline.strip():
        with st.spinner("Analyzing the headline..."):
            processed = preprocess(headline)
            vect = vectorizer.transform([processed])
            prediction = model.predict(vect)[0]

        # Show result beautifully
        if prediction == 0:
            st.success("✅ The news appears to be **REAL**.")
        else:
            st.error("🚨 This headline is likely **FAKE**.")
    else:
        st.warning("Please enter a headline before predicting.")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray; font-size:14px;'>
        Developed by <b>Arya Kshirsagar</b> — Zeal College of Engineering & Research (AIML)
    </p>
    """,
    unsafe_allow_html=True
)

