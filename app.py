import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (Streamlit Cloud needs this)
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

def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stopwords_set]
    return " ".join(words)

st.title("📰 Fake News Detector")
st.write("Enter a news headline to check if it is **Real** or **Fake**.")

user_input = st.text_area("Enter headline:", "")

if st.button("Predict"):
    if user_input.strip():
        processed = preprocess(user_input)
        vect = vectorizer.transform([processed])
        prediction = model.predict(vect)[0]

        if prediction == 0:
            st.success("✅ The news seems **Real**")
        else:
            st.error("🚨 The news is **Fake**")
    else:
        st.warning("Please enter some text before clicking Predict.")
