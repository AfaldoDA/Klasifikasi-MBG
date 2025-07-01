# app.py
import streamlit as st
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import demoji
from transformers import pipeline
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load kamus alay
@st.cache_data
def load_kamus_alay():
    try:
        kamus_df = pd.read_csv("kamus-alay.csv", encoding='latin1')
        if 'mbg' not in kamus_df['slang'].values:
            st.info("")
        else:
            st.warning(f"Kata 'mbg' masih ada di kamus alay: {kamus_df[kamus_df['slang'] == 'mbg']}")
        return dict(zip(kamus_df['slang'], kamus_df['formal']))
    except FileNotFoundError:
        st.error("File 'kamus-alay.csv' tidak ditemukan!")
        return {}

kamus = load_kamus_alay()
if not kamus:
    st.stop()

# Load model dari .pkl
@st.cache_resource
def load_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        model = saved_data['model']
        tokenizer = saved_data['tokenizer']
        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    except FileNotFoundError:
        st.error("File 'sentiment_model.pkl' tidak ditemukan! Jalankan save_model_to_pkl.py terlebih dahulu.")
        return None

sentiment_analysis = load_model()
if sentiment_analysis is None:
    st.stop()

# Fungsi pembersihan teks
def normalisasi(text):
    return ' '.join([kamus.get(kata, kata) for kata in text.split()])

def remove_emoji(text):
    return demoji.replace(text, "")

def bersihkan(text):
    if not text or pd.isna(text):
        return ""
    text = text.lower()
    text = remove_emoji(text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalisasi(text)
    text = stemmer.stem(text)
    return text

# Fungsi prediksi sentimen
def predict_sentiment(text):
    if not text.strip():
        return "neutral", 0.0
    result = sentiment_analysis(text)
    return result[0]['label'], result[0]['score']

# Antarmuka Streamlit
st.title("Prediksi Sentimen Kalimat")
st.write("Masukkan kalimat dalam bahasa Indonesia untuk menganalisis apakah sentimennya positif, netral, atau negatif.")

# Input kalimat
kalimat = st.text_area("Masukkan kalimat:", height=100)

if st.button("Prediksi"):
    if kalimat:
        # Bersihkan kalimat
        kalimat_bersih = bersihkan(kalimat)
        st.write("**Kalimat setelah dibersihkan:**", kalimat_bersih)

        # Prediksi sentimen
        # label, skor = predict_sentiment(kalimat_bersih)
        # st.write("**Hasil Prediksi:**")
        # st.write(f"**Sentimen:** {label}")
        # st.write(f"**Skor Kepercayaan:** {skor:.2f}")
        # Prediksi sentimen 
        label, skor = predict_sentiment(kalimat_bersih)

        # Tentukan warna berdasarkan label
        if label.lower() == "positive" or label.lower() == "positif":
            bg_color = "#d4edda"  # hijau muda
            text_color = "#155724"
        elif label.lower() == "negative" or label.lower() == "negatif":
            bg_color = "#f8d7da"  # merah muda
            text_color = "#721c24"
        else:
            bg_color = "#ffffff"  # putih
            text_color = "#000000"

        st.markdown(f"""
        <div style="background-color:{bg_color}; color:{text_color}; padding: 20px; border-radius: 10px;">
            <h4>Hasil Prediksi</h4>
            <p><strong>Sentimen:</strong> {label}</p>
            <p><strong>Skor Kepercayaan:</strong> {skor:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("Silakan masukkan kalimat terlebih dahulu!")
