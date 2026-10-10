import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import librosa

try:
    from moviepy.editor import AudioFileClip, VideoClip, concatenate_videoclips, VideoFileClip
except ImportError:
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.VideoClip import VideoClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    from moviepy.video.io.VideoFileClip import VideoFileClip

st.set_page_config(page_title="Suno AI Music Studio", page_icon="🎵", layout="wide")

st.title("🎵 Suno AI Studio & Video Renderer")
st.write("Studio pembuatan video musik, lirik, dan Shorts otomatis.")

# Sidebar API Key
st.sidebar.header("🔑 API Key")
api_key = st.sidebar.text_input("Gemini API Key:", type="password")
if api_key:
    genai.configure(api_key=api_key)

def generate_ai_response(prompt_text):
    for model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]:
        try:
            model = genai.GenerativeModel(model_name)
            return model.generate_content(prompt_text).text
        except Exception:
            continue
    return "❌ Gagal memanggil API Gemini."

# Input Utama
st.divider()
tema_lagu = st.text_input("Tema / Ide Lagu:", placeholder="Contoh: Misteri mistis malam jumat")
genre_lagu = st.multiselect("Genre:", ["Pop", "Rock", "Metal", "Dangdut Modern", "Mistis"], default=["Pop", "Dangdut Modern"])
genre_str = ", ".join(genre_lagu)

if st.button("✨ Buat Lirik Lagu"):
    if tema_lagu:
        with st.spinner("Membuat lirik..."):
            st.session_state["lirik"] = generate_ai_response(f"Buatkan lirik lagu dengan tema '{tema_lagu}' dan genre '{genre_str}'. Gunakan tag [Verse 1], [Chorus].")

lirik_input = st.text_area("Lirik Lagu:", value=st.session_state.get("lirik", "[Verse 1]\nTulis lirik di sini..."), height=150)

# Render Video
st.divider()
st.subheader("🎬 Render Video")
format_video = st.radio("Format:", ["📺 Long Video (16:9)", "📱 Short Video (9:16)"], horizontal=True)

uploaded_audio = st.file_uploader("Upload Audio (.mp3/.wav):", type=["mp3", "wav"])
uploaded_bg1 = st.file_uploader("Upload Background Utama:", type=["jpg", "jpeg", "png"])

if st.button("🔥 Render Video Sekarang", type="primary"):
    if not uploaded_audio or not uploaded_bg1:
        st.warning("Upload Audio dan Background terlebih dahulu!")
    else:
        with st.spinner("Merender video..."):
            try:
                audio_path = "temp_audio.mp3"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())

                audio_clip_full = AudioFileClip(audio_path)
                is_short = "Short" in format_video
                duration = min(30.0, audio_clip_full.duration) if is_short else audio_clip_full.duration
                audio_clip = audio_clip_full.subclip(0, duration)

                fps = 24
                target_w = 1080 if is_short else 1280
                target_h = 1920 if is_short else 720
                num_bars = 36 if is_short else 48

                y, sr = librosa.load(audio_path, sr=22050, duration=duration)
                hop_length = int(sr / fps)
                stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
                freq_bins = np.linspace(0, stft.shape[0] // 2, num_bars + 1, dtype=int)

                base_img = Image.open(uploaded_bg1).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)

                def make_frame(t):
                    frame_idx = min(int(t * fps), stft.shape[1] - 1)
                    canvas = base_img.copy()
                    draw = ImageDraw.Draw(canvas)

                    # Spektrum sederhana
                    amps = [np.mean(stft[freq_bins[b]:freq_bins[b+1], frame_idx]) for b in range(num_bars)]
                    max_amp = np.max(amps) if np.max(amps) > 0 else 1.0
                    norm_amps = [min(a / (max_amp * 0.7 + 1e-5), 1.0) for a in amps]

                    bar_w = 14 if is_short else 16
                    bar_gap = 5 if is_short else 6
                    base_y = target_h - 150
                    total_width = num_bars * (bar_w + bar_gap)
                    start_x = (target_w - total_width) // 2

                    for b in range(num_bars):
                        h_val = int(norm_amps[b] * 120) + 4
                        x0 = start_x + b * (bar_w + bar_gap)
                        draw.rectangle([x0, base_y - h_val, x0 + bar_w, base_y], fill=(0, 220, 255))

                    # Teks Hook / Info sederhana agar aman dari error
                    font = ImageFont.load_default()
                    if is_short and t <= 4.0:
                        draw.rectangle([40, 130, target_w - 40, 220], fill=(0, 0, 0, 200))
                        draw.text((70, 160), "🔥 Nonton Sampai Habis!", fill=(255, 230, 0), font=font)

                    return np.array(canvas)

                video_clip = VideoClip(make_frame, duration=duration)
                video_clip = video_clip.set_audio(audio_clip)

                output_path = "output_visualizer.mp4"
                video_clip.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)

                st.success("🎉 Render Selesai!")
                st.video(output_path)
            except Exception as e:
                st.error(f"Gagal merender: {e}")
