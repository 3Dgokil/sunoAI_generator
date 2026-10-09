import streamlit as st
import google.generativeai as genai
import os
import numpy as np
from PIL import Image, ImageDraw
import librosa
from moviepy.editor import AudioFileClip, VideoClip

# ----------------------------------------------------
# CONFIGURASI HALAMAN STREAMLIT
# ----------------------------------------------------
st.set_page_config(
    page_title="Suno AI Music Production Studio",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Suno AI All-in-One Studio & Video Renderer")
st.write("Aplikasi lengkap untuk membuat Lirik, Prompt Visual, SEO, dan Render Video Visualizer Sinematik!")

# ----------------------------------------------------
# MANAGEMENT GEMINI API KEY & MODEL FALLBACK
# ----------------------------------------------------
st.sidebar.header("🔑 Pengaturan API Key")
api_key = st.sidebar.text_input("Masukkan Gemini API Key:", type="password")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.sidebar.warning("Masukkan Gemini API Key untuk menggunakan fitur AI.")

def generate_ai_response(prompt_text):
    """
    Menggunakan model Gemini aktif dengan mekanisme fallback otomatis
    untuk menghindari 404 NotFound Error.
    """
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-1.0-pro"
    ]
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_text)
            return response.text
        except Exception:
            continue
    return "❌ Gagal memanggil API Gemini. Pastikan API Key benar dan memiliki kuota yang cukup."

# ----------------------------------------------------
# 1. INPUT TEMA & INSTRUMEN / GENRE
# ----------------------------------------------------
st.divider()
st.subheader("🎯 1. Tema & Genre Lagu")

col_t1, col_t2 = st.columns(2)
with col_t1:
    tema_lagu = st.text_input("Tema / Ide Utama Lagu:", placeholder="Contoh: Penyesalan cinta di malam hari")
with col_t2:
    genre_lagu = st.multiselect(
        "Pilih Genre / Gaya Musik Suno:",
        ["Pop", "Rock", "EDM", "Indie", "R&B", "Acoustic", "Cinematic", "Synthwave", "Lo-Fi", "Metal", "Jazz", "Tropical House"],
        default=["Pop", "Acoustic"]
    )

genre_str = ", ".join(genre_lagu)

# ----------------------------------------------------
# 2. GENERATOR JUDUL DINAMIS
# ----------------------------------------------------
st.divider()
st.subheader("💡 2. Generator Judul Lagu")

judul_terpilih = "Judul Lagu"
if st.button("✨ Buat Ide Judul Lagu"):
    if not tema_lagu:
        st.warning("Masukkan tema lagu terlebih dahulu!")
    else:
        prompt_judul = f"Buatkan 5 pilihan judul lagu yang sangat menarik, catchy, dan estetik berdasarkan tema: '{tema_lagu}' dan genre: '{genre_str}'. Tampilkan dalam bentuk daftar bernomor saja tanpa penjelasan tambahan."
        hasil_judul = generate_ai_response(prompt_judul)
        st.write(hasil_judul)

judul_terpilih = st.text_input("Ketik/Pilih Judul Lagu Utama:", value=tema_lagu if tema_lagu else "Lagu Tanpa Judul")

# ----------------------------------------------------
# 3. GENERATOR LIRIK HIGH-RETENTION (SUNO FORMAT)
# ----------------------------------------------------
st.divider()
st.subheader("✍️ 3. Generator Lirik Lagu (Suno AI Format)")

if st.button("🔥 Buat Lirik Lagu Lengkap", type="primary"):
    if not tema_lagu:
        st.warning("Masukkan tema lagu terlebih dahulu!")
    else:
        with st.spinner("Membuat lirik lagu sinematik high-retention..."):
            prompt_lirik = f"""
            Buatkan lirik lagu yang sangat emosional, catchy, dan berdaya simpan tinggi (high-retention) untuk Suno AI.
            Judul: {judul_terpilih}
            Tema: {tema_lagu}
            Genre/Style: {genre_str}

            Gunakan struktur tag meta Suno AI yang tepat seperti:
            [Verse 1]
            [Pre-Chorus]
            [Chorus]
            [Verse 2]
            [Bridge]
            [Guitar Solo] / [Drop]
            [Chorus]
            [Outro]
            [Fade Out]

            Pastikan rima dan ketukan liriknya pas untuk dinyanyikan AI.
            """
            lirik_res = generate_ai_response(prompt_lirik)
            st.text_area("Lirik Lagu Terbentuk (Tinggal Copas ke Suno):", value=lirik_res, height=32
