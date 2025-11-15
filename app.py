import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Required on Streamlit Cloud
nltk.download("stopwords")

ps = PorterStemmer()
stopwords_set = set(stopwords.words("english"))

# Load model & vectorizer
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()

# Preprocessing
def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stopwords_set]
    return " ".join(words)

# ---------------- Dashboard UI ---------------- #

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Sidebar
st.sidebar.title("🧭 Navigation")
st.sidebar.write("Use this dashboard to detect fake news using an ML model.")
st.sidebar.markdown("---")
st.sidebar.info("Created by **Arya Kshirsagar**\n\nB.Tech AIML | Zeal College")

# Main Title
st.markdown(
    """
    <h1 style='text-align:center; color:white;'>📰 Fake News Detection Dashboard</h1>
    <p style='text-align:center; font-size:18px; color:#cccccc;'>
        A Machine Learning-powered tool to classify news headlines as <b style="color:lightgreen;">Real</b> or 
        <b style="color:#ff4b4b;">Fake</b>.
    </p>
    <br>
    """, unsafe_allow_html=True
)

# Layout (Dashboard style)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✏️ Enter a news headline")
    headline = st.text_area(" ", height=150)

    if st.button("🔍 Run Fake News Detection", use_container_width=True):

        if headline.strip():

            with st.spinner("Analyzing… Please wait"):
                processed = preprocess(headline)
                vect = vectorizer.transform([processed])
                prediction = model.predict(vect)[0]

            if prediction == 0:
                st.success("✅ **REAL NEWS** — This headline appears trustworthy.")
            else:
                st.error("🚨 **FAKE NEWS** — This headline is likely misleading.")

        else:
            st.warning("Please enter a headline before predicting.")

with col2:
    st.subheader("📊 Model Info")
    st.info("""
    **Model:** Logistic Regression  
    **Vectorizer:** TF-IDF (5000 features)  
    **Accuracy:** ~91%  
    **Dataset:** WELFake News Dataset  
    """)

    st.subheader("ℹ️ How It Works")
    st.write("""
    - Removes stopwords  
    - Applies stemming  
    - Converts text to TF-IDF  
    - Logistic Regression makes the final prediction  
    """)

# Footer
st.markdown(
    """
    <br><hr>
    <p style='text-align:center; font-size:13px; color:gray;'>
        Built by <b>Arya Kshirsagar</b> — Fake News Detection Project (AIML)
    </p>
    """,
    unsafe_allow_html=True
)


