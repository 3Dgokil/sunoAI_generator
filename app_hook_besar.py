import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import librosa

# Import MoviePy dengan kompatibilitas versi 1.x & 2.x
try:
    from moviepy.editor import AudioFileClip, VideoClip, concatenate_videoclips, VideoFileClip
except ImportError:
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.VideoClip import VideoClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    from moviepy.video.io.VideoFileClip import VideoFileClip

# ----------------------------------------------------
# CONFIGURASI HALAMAN STREAMLIT
# ----------------------------------------------------
st.set_page_config(
    page_title="Suno AI Music Production Studio",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Suno AI All-in-One Studio & Video Renderer")
st.write("Studio lengkap dengan Lirik Berjalan, Smart Hook 30 Detik untuk Short, Multi-Artwork, Genre Spesial (Dangdut Modern, Mistis, Metal), & Video Merger!")

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
    tema_lagu = st.text_input("Tema / Ide Utama Lagu:", placeholder="Contoh: Keseruan malam minggu / Misteri mistis malam jumat")
with col_t2:
    genre_lagu = st.multiselect(
        "Pilih Genre / Gaya Musik Suno:",
        [
            "Pop", "Rock", "EDM", "Indie", "R&B", "Acoustic", "Cinematic", 
            "Synthwave", "Lo-Fi", "Metal", "Jazz", "Tropical House",
            "Dangdut Modern", "Mistis", "Heavy Metal"
        ],
        default=["Pop", "Dangdut Modern"]
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

lirik_state_key = "generated_lirik_text"
if lirik_state_key not in st.session_state:
    st.session_state[lirik_state_key] = "[Verse 1]\nTulis atau generate lirik di sini..."

if st.button("🔥 Buat Lirik Lagu Lengkap", type="primary"):
    if not tema_lagu:
        st.warning("Masukkan tema lagu terlebih dahulu!")
    else:
        with st.spinner("Membuat lirik lagu sinematik high-retention..."):
            prompt_lirik = f"Buatkan lirik lagu yang emosional, catchy, dan high-retention untuk Suno AI dengan Judul: '{judul_terpilih}', Tema: '{tema_lagu}', Genre: '{genre_str}'. Gunakan struktur tag meta Suno AI [Verse 1], [Chorus], [Bridge], [Outro]."
            lirik_res = generate_ai_response(prompt_lirik)
            st.session_state[lirik_state_key] = lirik_res

lirik_input = st.text_area("Lirik Lagu (Digunakan untuk Lirik Berjalan & Hook Otomatis):", value=st.session_state[lirik_state_key], height=220)

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
    prompt_img = f"Buatkan 3 prompt gambar terperinci berbeda dalam bahasa Inggris untuk Midjourney / Leonardo AI yang menggambarkan 3 adegan berbeda dari lagu berjudul '{judul_terpilih}' dengan tema '{tema_lagu}' dan nuansa genre '{genre_str}'."
    img_res = generate_ai_response(prompt_img)
    st.text_area("Prompt 3 Gambar Artwork (Bahasa Inggris):", value=img_res, height=130)

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
# 7. MESIN RENDER VIDEO PRO ULTIMATE
# ----------------------------------------------------
st.divider()
st.subheader("🎬 7. Mesin Render Video Pro (Long & Short Ultimate)")

st.write("Render video lengkap dengan lirik berjalan, Smart Hook 30 detik untuk Short, multi-artwork, dan opsi tanpa spektrum!")

format_video = st.radio("Pilih Format Video:", ["📺 Long Video (16:9 Horizontal + Lirik Berjalan)", "📱 Short Video (9:16 Vertikal + Smart Hook 30 Detik)"], horizontal=True)

col_v1, col_v2, col_v3, col_v4 = st.columns(4)

with col_v1:
    uploaded_audio = st.file_uploader("1. Upload Audio (.mp3/.wav):", type=["mp3", "wav"])

with col_v2:
    uploaded_bg1 = st.file_uploader("2. Artwork Utama / 1:", type=["jpg", "jpeg", "png"])

with col_v3:
    if "Short" in format_video:
        uploaded_bg2 = st.file_uploader("3. Artwork 2 (Opsional):", type=["jpg", "jpeg", "png"])
    else:
        uploaded_bg2 = None

with col_v4:
    if "Short" in format_video:
        uploaded_bg3 = st.file_uploader("4. Artwork 3 (Opsional):", type=["jpg", "jpeg", "png"])
    else:
        uploaded_bg3 = None

uploaded_logo = st.file_uploader("Upload Logo PNG Transparan (Opsional):", type=["png"])

col_opt1, col_opt2, col_opt3 = st.columns(3)
with col_opt1:
    style_spectrum = st.selectbox(
        "Gaya Spektrum Audio:",
        ["🚫 Tanpa Spektrum (Hanya Background + Logo)", "Balok (Equalizer)", "Bar (Solid)", "Line (Garis Wave)", "CLine (Circle Line)", "P2P (Point to Point)"]
    )
with col_opt2:
    color_theme = st.selectbox(
        "Warna Spektrum / Aksen:",
        ["Neon Cyan", "Sunset Red", "Electric Purple", "Cyber Green", "Gold Sunset"]
    )
with col_opt3:
    mode_render = st.radio("Pilih Mode Render:", ["🧪 Test Preview (3 Detik Cepat)", "🎬 Render Full Video"], horizontal=True)

cta_text = ""
if "Short" in format_video:
    cta_text = st.text_input("Teks CTA (Muncul di Akhir Video Short):", value="Dengerin versi full-nya & Subscribe ya! 🎵")

if st.button("🔥 Render Video Sekarang", type="primary"):
    if uploaded_audio is None or uploaded_bg1 is None:
        st.warning("Mohon unggah file Audio dan minimal Artwork 1 terlebih dahulu!")
    else:
        with st.spinner("Menganalisis hook terbaik & merender video..."):
            try:
                audio_path = "temp_audio.mp3"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())

                audio_clip_full = AudioFileClip(audio_path)
                is_short = "Short" in format_video

                if "3 Detik" in mode_render:
                    duration = min(3.0, audio_clip_full.duration)
                    audio_clip = audio_clip_full.subclip(0, duration)
                    start_time = 0
                elif is_short:
                    target_short_dur = 30.0
                    if audio_clip_full.duration > target_short_dur:
                        start_time = min(15.0, audio_clip_full.duration - target_short_dur)
                        if start_time < 0: 
                            start_time = 0
                        duration = min(target_short_dur, audio_clip_full.duration - start_time)
                    else:
                        start_time = 0
                        duration = audio_clip_full.duration
                    
                    audio_clip = audio_clip_full.subclip(start_time, start_time + duration)
                else:
                    duration = audio_clip_full.duration
                    audio_clip = audio_clip_full
                    start_time = 0

                fps = 24
                target_w = 1080 if is_short else 1280
                target_h = 1920 if is_short else 720
                num_bars = 36 if is_short else 48

                y, sr = librosa.load(audio_path, sr=22050, offset=start_time, duration=duration)
                hop_length = int(sr / fps)
                stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
                freq_bins = np.linspace(0, stft.shape[0] // 2, num_bars + 1, dtype=int)

                raw_lines = lirik_input.split('\n')
                clean_lyrics = []
                for line in raw_lines:
                    line_str = line.strip()
                    if line_str and not line_str.startswith('['):
                        clean_lyrics.append(line_str)
                if not clean_lyrics:
                    clean_lyrics = [judul_terpilih, "Nikmati alunan musik ini..."]

                chorus_idx = len(clean_lyrics) // 2 if len(clean_lyrics) > 2 else 0
                auto_hook_text = clean_lyrics[chorus_idx] if len(clean_lyrics) > 0 else "Dengarkan lagu ini sampai habis..."

                bg_images = []
                img_raw_1 = Image.open(uploaded_bg1).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                bg_images.append(img_raw_1)

                if uploaded_bg2:
                    img_raw_2 = Image.open(uploaded_bg2).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                    bg_images.append(img_raw_2)
                else:
                    bg_images.append(img_raw_1)

                if uploaded_bg3:
                    img_raw_3 = Image.open(uploaded_bg3).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                    bg_images.append(img_raw_3)
                else:
                    bg_images.append(img_raw_1)

                logo_img = None
                if uploaded_logo:
                    try:
                        logo_img = Image.open(uploaded_logo).convert("RGBA")
                        logo_img.thumbnail((120 if is_short else 140, 120 if is_short else 140), Image.Resampling.LANCZOS)
                    except Exception:
                        logo_img = None

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

                def make_frame(t):
                    frame_idx = min(int(t * fps), stft.shape[1] - 1)
                    progress = t / duration

                    if is_short and len(bg_images) >= 3:
                        if progress < 0.33:
                            active_bg = bg_images[0]
                        elif progress < 0.66:
                            active_bg = bg_images[1]
                        else:
                            active_bg = bg_images[2]
                    else:
                        active_bg = bg_images[0]

                    scale = 1.0 + (0.10 * progress)
                    crop_w = int(target_w / scale)
                    crop_h = int(target_h / scale)
                    left = (target_w - crop_w) // 2
                    top = (target_h - crop_h) // 2
                    
                    frame_bg = active_bg.crop((left, top, left + crop_w, top + crop_h)).resize((target_w, target_h), Image.Resampling.LANCZOS)
                    canvas = frame_bg.copy()
                    draw = ImageDraw.Draw(canvas)

                    amps = [np.mean(stft[freq_bins[b]:freq_bins[b+1], frame_idx]) for b in range(num_bars)]
                    max_amp = np.max(amps) if np.max(amps) > 0 else 1.0
                    norm_amps = [min(a / (max_amp * 0.7 + 1e-5), 1.0) for a in amps]

                    if style_spectrum != "🚫 Tanpa Spektrum (Hanya Background + Logo)":
                        bar_w = 14 if is_short else 16
                        bar_gap = 5 if is_short else 6
                        base_y = target_h - (180 if is_short else 90)
                        max_bar_h = 140 if is_short else 180
                        
                        total_width = num_bars * (bar_w + bar_gap)
                        start_x = (target_w - total_width) // 2

                        if style_spectrum == "Balok (Equalizer)":
                            for b in range(num_bars):
                                h_val = int(norm_amps[b] * max_bar_h) + 6
                                x0 = start_x + b * (bar_w + bar_gap)
                                x1 = x0 + bar_w
                                num_segments = 10
                                seg_h = max(h_val // num_segments, 2)
                                for s in range(num_segments):
                                    seg_progress = s / num_segments
                                    y1_seg = base_y - (s * (seg_h + 2))
                                    y0_seg = y1_seg - seg_h
                                    if y0_seg < base_y - h_val: 
                                        break
                                    color = primary_color if seg_progress < 0.5 else (secondary_color if seg_progress < 0.8 else accent_color)
                                    draw.rectangle([x0, y0_seg, x1, y1_seg], fill=color)

                        elif style_spectrum == "Bar (Solid)":
                            for b in range(num_bars):
                                h_val = int(norm_amps[b] * max_bar_h) + 4
                                x0 = start_x + b * (bar_w + bar_gap)
                                x1 = x0 + bar_w
                                y0 = base_y - h_val
                                draw.rectangle([x0, y0, x1, base_y], fill=primary_color)

                        elif style_spectrum == "Line (Garis Wave)":
                            points = []
                            for b in range(num_bars):
                                x = start_x + b * (bar_w + bar_gap) + (bar_w // 2)
                                y = base_y - int(norm_amps[b] * max_bar_h)
                                points.append((x, y))
                            if len(points) > 1:
                                draw.line(points, fill=primary_color, width=4)

                        elif style_spectrum == "CLine (Circle Line)":
                            cx, cy = target_w // 2, (target_h // 2) + (100 if is_short else 0)
                            base_r = 120
                            circle_points = []
                            for b in range(num_bars):
                                angle = (2 * math.pi / num_bars) * b
                                r = base_r + (norm_amps[b] * 70)
                                px = cx + int(r * math.cos(angle))
                                py = cy + int(r * math.sin(angle))
                                circle_points.append((px, py))
                            if len(circle_points) > 1:
                                circle_points.append(circle_points[0])
                                draw.line(circle_points, fill=primary_color, width=4)

                        elif style_spectrum == "P2P (Point to Point)":
                            points = []
                            for b in range(num_bars):
                                x = start_x + b * (bar_w + bar_gap) + (bar_w // 2)
                                y = base_y - int(norm_amps[b] * max_bar_h)
                                points.append((x, y))
                                draw.ellipse([x-4, y-4, x+4, y+4], fill=accent_color)
                            if len(points) > 1:
                                draw.line(points, fill=primary_color, width=3)

                    if logo_img:
                        avg_bass = np.mean(norm_amps[:num_bars//4])
                        pulse_scale = 1.0 + (0.15 * avg_bass)
                        lw, lh = logo_img.size
                        new_lw, new_lh = int(lw * pulse_scale), int(lh * pulse_scale)
                        pulsed_logo = logo_img.resize((new_lw, new_lh), Image.Resampling.LANCZOS)
                        canvas.paste(pulsed_logo, (40, 40), pulsed_logo)

                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None

                    if not is_short:
                        if clean_lyrics:
                            lyric_index = int(progress * len(clean_lyrics)) % len(clean_lyrics)
                            active_lyric = clean_lyrics[lyric_index]
                            box_y = 570
                            draw.rectangle([100, box_y, target_w - 100, box_y + 55], fill=(0, 0, 0))
                            draw.text((120, box_y + 18), f"♫ {active_lyric}", fill=(255, 255, 255), font=font)
                    else:
                        # HOOK SHORT: kotak besar dan teks tebal yang terbaca di layar HP.
                        if t <= 4.0 and auto_hook_text:
                            box_x1 = 35
                            box_y1 = 100
                            box_x2 = target_w - 35
                            box_y2 = 430
                            max_text_width = box_x2 - box_x1 - 54
                            max_text_height = box_y2 - box_y1 - 48

                            hook_font_paths = [
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
                                "/system/fonts/Roboto-Bold.ttf",
                            ]

                            def load_hook_font(size):
                                for font_path in hook_font_paths:
                                    try:
                                        return ImageFont.truetype(font_path, size)
                                    except (OSError, IOError):
                                        pass
                                try:
                                    return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
                                except (OSError, IOError):
                                    return ImageFont.load_default()

                            hook_words = str(auto_hook_text).strip().split()
                            hook_lines = []
                            hook_font = load_hook_font(56)

                            # Pilih ukuran font besar yang muat, lalu bungkus teks otomatis.
                            for font_size in range(56, 27, -2):
                                hook_font = load_hook_font(font_size)
                                hook_lines = []
                                current_line = ""

                                for word in hook_words:
                                    candidate = (current_line + " " + word).strip()
                                    candidate_box = draw.textbbox((0, 0), candidate, font=hook_font)
                                    if candidate_box[2] - candidate_box[0] <= max_text_width:
                                        current_line = candidate
                                    else:
                                        if current_line:
                                            hook_lines.append(current_line)
                                        current_line = word

                                if current_line:
                                    hook_lines.append(current_line)

                                # Jika satu kata sangat panjang, potong ukuran font masih dicoba.
                                line_boxes = [
                                    draw.textbbox((0, 0), line, font=hook_font)
                                    for line in hook_lines
                                ]
                                line_heights = [
                                    max(1, b[3] - b[1]) for b in line_boxes
                                ]
                                total_text_height = sum(line_heights) + max(0, len(hook_lines) - 1) * 12

                                if (
                                    hook_lines
                                    and max((b[2] - b[0] for b in line_boxes), default=0) <= max_text_width
                                    and total_text_height <= max_text_height
                                ):
                                    break

                            # Kotak hitam pekat, bingkai neon cyan, agar kontras dengan artwork.
                            draw.rounded_rectangle(
                                [box_x1, box_y1, box_x2, box_y2],
                                radius=24,
                                fill=(0, 0, 0),
                                outline=(0, 220, 255),
                                width=5,
                            )

                            line_boxes = [
                                draw.textbbox((0, 0), line, font=hook_font)
                                for line in hook_lines
                            ]
                            line_heights = [
                                max(1, b[3] - b[1]) for b in line_boxes
                            ]
                            total_text_height = sum(line_heights) + max(0, len(hook_lines) - 1) * 12
                            current_y = box_y1 + (box_y2 - box_y1 - total_text_height) // 2

                            for line, line_box, line_height in zip(hook_lines, line_boxes, line_heights):
                                line_width = line_box[2] - line_box[0]
                                text_x = (target_w - line_width) // 2
                                draw.text(
                                    (text_x, current_y - line_box[1]),
                                    line,
                                    fill=(255, 255, 255),
                                    font=hook_font,
                                    stroke_width=2,
                                    stroke_fill=(0, 0, 0),
                                )
                                current_y += line_height + 12

                        # CTA akhir Short tetap tersedia; bagian ini hanya melengkapi
                        # blok akhir file yang terpotong pada file sumber.
                        if t >= (duration - 4.0) and cta_text:
                            cta_box = [50, target_h - 280, target_w - 50, target_h - 180]
                            draw.rounded_rectangle(
                                cta_box,
                                radius=18,
                                fill=(20, 20, 20),
                                outline=primary_color,
                                width=3,
                            )
                            cta_font = font
                            for font_size in (34, 30, 26, 22):
                                cta_font = load_hook_font(font_size) if "load_hook_font" in locals() else font
                                try:
                                    cta_bbox = draw.textbbox((0, 0), cta_text, font=cta_font)
                                    if cta_bbox[2] - cta_bbox[0] <= target_w - 140:
                                        break
                                except Exception:
                                    cta_font = font
                                    break
                            draw.text(
                                (target_w // 2, target_h - 230),
                                cta_text,
                                fill=(255, 255, 255),
                                font=cta_font,
                                anchor="mm",
                            )

                    return np.asarray(canvas)

                # Buat video dari frame dan pasangkan audio.
                video_clip = VideoClip(make_frame, duration=duration)
                try:
                    video_clip = video_clip.set_audio(audio_clip)
                except AttributeError:
                    video_clip = video_clip.with_audio(audio_clip)

                output_path = "hasil_render_short.mp4" if is_short else "hasil_render_long.mp4"
                video_clip.write_videofile(
                    output_path,
                    fps=fps,
                    codec="libx264",
                    audio_codec="aac",
                    preset="ultrafast" if "3 Detik" in mode_render else "medium",
                    threads=2,
                    logger=None,
                )

                st.success(f"Render selesai: {output_path}")
                st.video(output_path)
                with open(output_path, "rb") as rendered_file:
                    st.download_button(
                        "⬇️ Download Video Hasil Render",
                        data=rendered_file.read(),
                        file_name=output_path,
                        mime="video/mp4",
                        type="primary",
                    )

                try:
                    video_clip.close()
                except Exception:
                    pass
                if audio_clip is not audio_clip_full:
                    try:
                        audio_clip.close()
                    except Exception:
                        pass
                try:
                    audio_clip_full.close()
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Render gagal: {e}")
