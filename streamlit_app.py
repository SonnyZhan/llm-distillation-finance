import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_models():
    return SentenceTransformer("all-MiniLM-L6-v2"), joblib.load("student.joblib")

encoder, student = load_models()

st.title("Tiny finance sentiment model")
st.caption("A small model trained only on labels written by Claude.")
sentence = st.text_input("Financial news sentence")

if sentence:
    probs = student.predict_proba(encoder.encode([sentence]))[0]
    for label, p in sorted(zip(student.classes_, probs), key=lambda x: -x[1]):
        st.write(f"**{label}**: {p:.0%}")
        st.progress(float(p))
