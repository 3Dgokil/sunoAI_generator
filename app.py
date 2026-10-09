import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import math
from PIL import Image, ImageDraw
import librosa

# Import MoviePy dengan kompatibilitas versi 1.x & 2.x
try:
    from moviepy.editor import AudioFileClip, VideoClip
except ImportError:
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.VideoClip import VideoClip

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
            prompt_lirik = f"Buatkan lirik lagu yang emosional, catchy, dan high-retention untuk Suno AI dengan Judul: '{judul_terpilih}', Tema: '{tema_lagu}', Genre: '{genre_str}'. Gunakan struktur tag meta Suno AI [Verse 1], [Chorus], [Bridge], [Outro]."
            lirik_res = generate_ai_response(prompt_lirik)
            st.text_area("Lirik Lagu Terbentuk (Tinggal Copas ke Suno):", value=lirik_res, height=320)

# ----------------------------------------------------
# 4. OPTIMIZER STYLE / PROMPT SUNO
# ----------------------------------------------------
st.divider()
st.subheader("🎛️ 4. Suno AI Style Prompt Optimizer")

if st.button("⚡ Buat Style Prompt Suno (Max 120 Karakter)"):
    prompt_style = f"Buatkan ringkasan Style / Prompt Musik untuk kolom Style of Music di Suno AI berdasarkan genre '{genre_str}' dan tema '{tema_lagu}'. Karakter maksimal 120 karakter, dalam bahasa Inggris, dipisahkan koma."
    style_res = generate_ai_response(prompt_style)
    st.code(style_res, language="text")

# ----------------------------------------------------
# 5. VISUAL PROMPT GENERATOR (MIDJOURNEY / LEONARDO)
# ----------------------------------------------------
st.divider()
st.subheader("🖼️ 5. Generator Prompt Gambar Background")

if st.button("🎨 Buat Visual Prompt Artwork"):
    prompt_img = f"Buatkan 1 prompt gambar terperinci dalam bahasa Inggris untuk Midjourney / Leonardo AI yang menggambarkan artwork sampul album lagu berjudul '{judul_terpilih}' dengan tema '{tema_lagu}'. Sertakan detail style sinematik, lighting, 8k resolution, tanpa teks."
    img_res = generate_ai_response(prompt_img)
    st.text_area("Prompt Gambar Artwork (Bahasa Inggris):", value=img_res, height=100)

# ----------------------------------------------------
# 6. SEO OPTIMIZER (DESKRIPSI & HASHTAG YOUTUBE/TIKTOK)
# ----------------------------------------------------
st.divider()
st.subheader("🚀 6. SEO Optimizer (YouTube & TikTok)")

if st.button("📱 Buat Deskripsi & Hashtag Viral"):
    prompt_seo = f"Buatkan deskripsi singkat YouTube/TikTok yang menarik dan 15 hashtag viral paling relevan untuk merilis lagu berjudul '{judul_terpilih}' dengan genre '{genre_str}'."
    seo_res = generate_ai_response(prompt_seo)
    st.write(seo_res)

# ----------------------------------------------------
# 7. MESIN RENDER VIDEO ADVANCED (CUSTOM SPECTRUM & COLOR)
# ----------------------------------------------------
st.divider()
st.subheader("🎬 7. Mesin Render Video Pro (Custom Visualizer & Warna)")

st.write("Buat video visualizer musik sinematik dengan spektrum audio otomatis untuk format video long (16:9), kustomisasi warna, dan logo dinamis!")

col_v1, col_v2, col_v3 = st.columns(3)

with col_v1:
    uploaded_audio = st.file_uploader("1. Upload Audio (.mp3 / .wav):", type=["mp3", "wav"])

with col_v2:
    uploaded_bg = st.file_uploader("2. Upload Background Artwork (.jpg / .png):", type=["jpg", "jpeg", "png"])

with col_v3:
    uploaded_logo = st.file_uploader("3. Upload Logo PNG Transparan (Opsional):", type=["png"])

col_opt1, col_opt2, col_opt3 = st.columns(3)
with col_opt1:
    style_spectrum = st.selectbox(
        "Gaya Spektrum Audio:",
        ["Balok (Equalizer)", "Bar (Solid)", "Line (Garis Wave)", "CLine (Circle Line)", "P2P (Point to Point)"]
    )
with col_opt2:
    color_theme = st.selectbox(
        "Warna Spektrum:",
        ["Neon Cyan", "Sunset Red", "Electric Purple", "Cyber Green", "Gold Sunset"]
    )
with col_opt3:
    mode_render = st.radio("Pilih Mode Render:", ["🧪 Test Preview (3 Detik Cepat)", "🎬 Render Full Video Lagu"], horizontal=True)

if st.button("🔥 Render Video Visualizer Sekarang", type="primary"):
    if uploaded_audio is None or uploaded_bg is None:
        st.warning("Mohon unggah file Audio dan Background Artwork terlebih dahulu!")
    else:
        with st.spinner("Sedang merender video visualizer di server Streamlit..."):
            try:
                # 1. Simpan File Audio Sementara
                audio_path = "temp_audio.mp3"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())

                # 2. Tentukan Durasi berdasarkan Opsi Test / Full
                audio_clip_full = AudioFileClip(audio_path)
                
                if "3 Detik" in mode_render:
                    duration = min(3.0, audio_clip_full.duration)
                    audio_clip = audio_clip_full.subclip(0, duration)
                else:
                    duration = audio_clip_full.duration
                    audio_clip = audio_clip_full

                fps = 24

                # OLEH KARENA VIDEO BERSKALA LONG (16:9 / 1280x720), HITUNG JUMLAH BAR OTOMATIS
                num_bars = 48  # Jumlah bar optimal untuk skala video long 1280px
                
                # 3. Analisis Frekuensi Audio dengan Librosa
                y, sr = librosa.load(audio_path, sr=22050, duration=duration)
                hop_length = int(sr / fps)
                stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
                freq_bins = np.linspace(0, stft.shape[0] // 2, num_bars + 1, dtype=int)

                # 4. Prepare Background & Logo
                bg_img = Image.open(uploaded_bg).convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
                bg_w, bg_h = bg_img.size

                logo_img = None
                if uploaded_logo:
                    try:
                        logo_img = Image.open(uploaded_logo).convert("RGBA")
                        logo_img.thumbnail((220, 220), Image.Resampling.LANCZOS)
                    except Exception:
                        logo_img = None

                # MAPPING WARNA SELEKSI
                def get_theme_colors(theme):
                    if theme == "Neon Cyan":
                        return (0, 220, 255), (0, 150, 255), (0, 255, 200)
                    elif theme == "Sunset Red":
                        return (255, 50, 80), (255, 120, 0), (255, 200, 0)
                    elif theme == "Electric Purple":
                        return (180, 0, 255), (255, 0, 180), (0, 200, 255)
                    elif theme == "Cyber Green":
                        return (0, 255, 120), (180, 255, 0), (0, 255, 220)
                    elif theme == "Gold Sunset":
                        return (255, 190, 0), (255, 100, 0), (255, 230, 100)
                    return (0, 255, 100), (255, 230, 0), (255, 30, 30)

                primary_color, secondary_color, accent_color = get_theme_colors(color_theme)

                # 5. Fungsi Generator Frame Video
                def make_frame(t):
                    frame_idx = min(int(t * fps), stft.shape[1] - 1)
                    progress = t / duration

                    # A. Efek Zoom In Background
                    scale = 1.0 + (0.12 * progress)
                    crop_w, crop_h = int(bg_w / scale), int(bg_h / scale)
                    left = (bg_w - crop_w) // 2
                    top = (bg_h - crop_h) // 2
                    
                    frame_bg = bg_img.crop((left, top, left + crop_w, top + crop_h)).resize((bg_w, bg_h), Image.Resampling.LANCZOS)
                    canvas = frame_bg.copy()
                    draw = ImageDraw.Draw(canvas)

                    # B. Amplitudo Audio per Frame
                    amps = [np.mean(stft[freq_bins[b]:freq_bins[b+1], frame_idx]) for b in range(num_bars)]
                    max_amp = np.max(amps) if np.max(amps) > 0 else 1.0
                    norm_amps = [min(a / (max_amp * 0.7 + 1e-5), 1.0) for a in amps]

                    # C. Pengaturan Ukuran Visualizer Otomatis Long Video
                    bar_w = 16
                    bar_gap = 6
                    base_y = bg_h - 70
                    max_bar_h = 180
                    
                    total_width = num_bars * (bar_w + bar_gap)
                    start_x = (bg_w - total_width) // 2

                    # --- MODE BALOK (Equalizer Bertumpuk) ---
                    if style_spectrum == "Balok (Equalizer)":
                        for b in range(num_bars):
                            h_val = int(norm_amps[b] * max_bar_h) + 6
                            x0 = start_x + b * (bar_w + bar_gap)
                            x1 = x0 + bar_w
                            num_segments = 12
                            seg_h = max(h_val // num_segments, 2)
                            
                            for s in range(num_segments):
                                seg_progress = s / num_segments
                                y1_seg = base_y - (s * (seg_h + 2))
                                y0_seg = y1_seg - seg_h
                                if y0_seg < base_y - h_val: 
                                    break
                                color = primary_color if seg_progress < 0.5 else (secondary_color if seg_progress < 0.8 else accent_color)
                                draw.rectangle([x0, y0_seg, x1, y1_seg], fill=color)

                    # --- MODE BAR (Batang Solid Modern) ---
                    elif style_spectrum == "Bar (Solid)":
                        for b in range(num_bars):
                            h_val = int(norm_amps[b] * max_bar_h) + 4
                            x0 = start_x + b * (bar_w + bar_gap)
                            x1 = x0 + bar_w
                            y0 = base_y - h_val
                            draw.rectangle([x0, y0, x1, base_y], fill=primary_color)

                    # --- MODE LINE (Garis Wave) ---
                    elif style_spectrum == "Line (Garis Wave)":
                        points = []
                        for b in range(num_bars):
                            x = start_x + b * (bar_w + bar_gap) + (bar_w // 2)
                            y = base_y - int(norm_amps[b] * max_bar_h)
                            points.append((x, y))
                        if len(points) > 1:
                            draw.line(points, fill=primary_color, width=5)

                    # --- MODE CLine (Circle Line) ---
                    elif style_spectrum == "CLine (Circle Line)":
                        cx, cy = bg_w // 2, bg_h // 2
                        base_r = 140
                        circle_points = []
                        for b in range(num_bars):
                            angle = (2 * math.pi / num_bars) * b
                            r = base_r + (norm_amps[b] * 90)
                            px = cx + int(r * math.cos(angle))
                            py = cy + int(r * math.sin(angle))
                            circle_points.append((px, py))
                        if len(circle_points) > 1:
                            circle_points.append(circle_points[0]) # Tutup Lingkaran
                            draw.line(circle_points, fill=primary_color, width=5)

                    # --- MODE P2P (Point to Point) ---
                    elif style_spectrum == "P2P (Point to Point)":
                        points = []
                        for b in range(num_bars):
                            x = start_x + b * (bar_w + bar_gap) + (bar_w // 2)
                            y = base_y - int(norm_amps[b] * max_bar_h)
                            points.append((x, y))
                            draw.ellipse([x-5, y-5, x+5, y+5], fill=accent_color)
                        if len(points) > 1:
                            draw.line(points, fill=primary_color, width=3)

                    # D. Denyut Logo (Bass Pulsing Sync)
                    if logo_img:
                        avg_bass = np.mean(norm_amps[:num_bars//4])
                        pulse_scale = 1.0 + (0.18 * avg_bass)
                        
                        lw, lh = logo_img.size
                        new_lw, new_lh = int(lw * pulse_scale), int(lh * pulse_scale)
                        
                        pulsed_logo = logo_img.resize((new_lw, new_lh), Image.Resampling.LANCZOS)
                        logo_x = (bg_w - new_lw) // 2
                        logo_y = (bg_h - new_lh) // 2 - 20
                        
                        canvas.paste(pulsed_logo, (logo_x, logo_y), pulsed_logo)

                    return np.array(canvas)

                # 6. Render dan Buat File Video MP4
                video_clip = VideoClip(make_frame, duration=duration)
                video_clip = video_clip.set_audio(audio_clip)

                output_path = "output_visualizer.mp4"
                video_clip.write_videofile(
                    output_path,
                    fps=fps,
                    codec="libx264",
                    audio_codec="aac",
                    preset="ultrafast",
                    logger=None
                )

                st.success("🎉 Render Selesai!")
                st.video(output_path)

                # Tombol Download Video
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Hasil Video MP4",
                        data=file,
                        file_name="Visualizer_Preview.mp4" if "3 Detik" in mode_render else f"{judul_terpilih}_Visualizer.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"Gagal merender video: {e}")
