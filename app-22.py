import streamlit as st
import json
import urllib.request
import urllib.error
import os
import re
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
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
# MANAGEMENT GEMINI API KEY DARI STREAMLIT SECRETS
# Menggunakan REST API resmi agar tidak bergantung pada SDK lama.
# ----------------------------------------------------
st.sidebar.header("🔑 Status Gemini API")

try:
    api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
except Exception:
    api_key = ""

if api_key:
    st.sidebar.success("Gemini API Key terbaca dari Secrets")
else:
    st.sidebar.error(
        "API Key belum ditemukan. Tambahkan GEMINI_API_KEY di "
        "Streamlit Cloud > App settings > Secrets."
    )


def generate_ai_response(prompt_text):
    """Memanggil Gemini REST API dan menampilkan detail error yang berguna."""
    if not api_key:
        return (
            "❌ API Key belum ditemukan. Periksa Settings > Secrets dan "
            "pastikan nama variabelnya GEMINI_API_KEY."
        )

    # Ambil daftar model langsung dari API agar tidak bergantung pada
    # nama model yang sudah tidak tersedia atau tidak didukung.
    last_error = "Tidak ada detail error dari Gemini."
    try:
        list_url = (
            "https://generativelanguage.googleapis.com/v1beta/models?key="
            + api_key
        )
        list_req = urllib.request.Request(
            list_url,
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(list_req, timeout=30) as response:
            model_data = json.loads(response.read().decode("utf-8"))
        models_to_try = []
        for item in model_data.get("models", []):
            methods = item.get("supportedGenerationMethods", [])
            name = item.get("name", "")
            if name.startswith("models/") and "generateContent" in methods:
                model_id = name.split("/", 1)[1]
                # Generator judul/lirik hanya memerlukan model teks.
                # Hindari model khusus gambar, audio, TTS, dan embedding.
                lowered = model_id.lower()
                if any(tag in lowered for tag in ("image", "imagen", "tts", "audio", "embedding")):
                    continue
                models_to_try.append(model_id)
        if not models_to_try:
            return (
                "❌ API Gemini tidak mengembalikan model yang mendukung "
                "generateContent. Periksa API key, akses model, dan konfigurasi project."
            )
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8", errors="replace"))
            detail = body.get("error", {}).get("message", f"HTTP {e.code}")
        except Exception:
            detail = f"HTTP {e.code}"
        st.error(f"Gagal mengambil daftar model Gemini: {detail}")
        return "❌ Tidak bisa membaca daftar model Gemini. Periksa API key dan akses API."
    except Exception as e:
        st.error(f"Gagal mengambil daftar model Gemini: {type(e).__name__}: {e}")
        return "❌ Tidak bisa menghubungi Gemini API."

    # Prioritaskan model teks Flash/Lite yang stabil; hindari model preview bila ada opsi stabil.
    def model_priority(name):
        n = name.lower()
        return (
            ("preview" in n or "experimental" in n),
            ("flash-lite" not in n and "flash_lite" not in n),
            ("flash" not in n),
            name,
        )

    models_to_try.sort(key=model_priority)

    for model_name in models_to_try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            + model_name
            + ":generateContent?key="
            + api_key
        )
        payload = {
            "contents": [{"parts": [{"text": str(prompt_text)}]}],
            "generationConfig": {"temperature": 0.8, "maxOutputTokens": 2048},
        }
        request_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))

            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [
                    part.get("text", "")
                    for part in parts
                    if isinstance(part, dict) and part.get("text")
                ]
                generated_text = "\n".join(text_parts).strip()
                if generated_text:
                    return generated_text

            last_error = f"{model_name}: respons kosong atau tidak berisi teks."

        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode("utf-8", errors="replace")
                error_json = json.loads(error_body)
                message = error_json.get("error", {}).get("message", error_body)
            except Exception:
                message = f"HTTP {e.code}"
            # URL tidak ditampilkan karena mengandung API key.
            last_error = f"{model_name}: HTTP {e.code} — {message}"
            # Jika satu model tidak punya kuota, lanjutkan mencoba model teks lain.
            # Error autentikasi/izin biasanya berlaku untuk seluruh project.
            if e.code in (401, 403):
                break
            if e.code == 429:
                continue

        except Exception as e:
            last_error = f"{model_name}: {type(e).__name__}: {e}"

    st.error(f"Gemini gagal merespons. Detail error: {last_error}")
    return (
        "❌ Gemini belum berhasil membuat respons. Periksa detail error "
        "di atas; jika error kuota atau API key, perbaiki di Google AI Studio."
    )

# ----------------------------------------------------
# 1. INPUT TEMA & INSTRUMEN / GENRE
# ----------------------------------------------------
st.divider()
st.subheader("🎯 1. Tema, Subtema & Genre Lagu")
st.caption("Isi tema utama. Subtema boleh dikosongkan untuk memberi AI ruang mengembangkan ide.")

col_t1, col_t2 = st.columns(2)
with col_t1:
    tema_lagu = st.text_input(
        "Tema Utama Lagu:",
        placeholder="Contoh: Tangis anak untuk ibu",
        help="Gagasan inti lagu. Contoh: perjuangan hidup, cinta, keluarga, kritik sosial."
    )
    subtema_lagu = st.text_input(
        "Subtema (opsional):",
        placeholder="Contoh: penyesalan, rindu, doa terakhir",
        help="Detail sudut cerita atau emosi. Boleh dikosongkan."
    )
with col_t2:
    genre_lagu = st.multiselect(
        "Pilih Genre / Gaya Musik Suno:",
        [
            "Pop", "Rock", "EDM", "Indie", "R&B", "Acoustic", "Cinematic",
            "Synthwave", "Lo-Fi", "Metal", "Jazz", "Tropical House",
            "Dangdut Modern", "Mistis", "Heavy Metal", "Dangdut Koplo",
            "Orchestral", "Folk", "Reggae", "Hip-Hop / Rap", "Punk Rock",
            "Gothic", "Phonk", "Ambient", "Blues", "Country"
        ],
        default=["Pop", "Dangdut Modern"],
        help="Pilih satu atau beberapa genre untuk menggabungkan gaya musik."
    )

genre_str = ", ".join(genre_lagu) if genre_lagu else "Pop"
subtema_context = subtema_lagu.strip() if subtema_lagu.strip() else "Tidak ditentukan; kembangkan secara kreatif dari tema utama."
tema_lengkap = f"Tema utama: {tema_lagu}. Subtema: {subtema_context}"

# ----------------------------------------------------
# 2. GENERATOR JUDUL DINAMIS
# ----------------------------------------------------
st.divider()
st.subheader("💡 2. Generator Judul Lagu")
col_j1, col_j2 = st.columns([1, 2])
with col_j1:
    target_pasar = st.selectbox(
        "Target pendengar / bahasa judul:",
        ["Lokal (Indonesia)", "Global (Internasional)", "Campuran (Lokal + Global)"],
        index=0,
        help="Lokal: relevan untuk pendengar Indonesia. Global: mudah dipahami pasar internasional. Campuran: judul Indonesia dan Inggris."
    )
with col_j2:
    jumlah_judul = st.selectbox("Jumlah ide judul:", [5, 10, 15], index=1)

if st.button("✨ Buat Ide Judul Lagu"):
    if not tema_lagu.strip():
        st.warning("Masukkan tema utama lagu terlebih dahulu!")
    else:
        if target_pasar == "Lokal (Indonesia)":
            arahan_target = "Gunakan bahasa Indonesia yang alami, kuat, mudah diingat, relevan dengan budaya dan pendengar Indonesia."
        elif target_pasar == "Global (Internasional)":
            arahan_target = "Gunakan bahasa Inggris yang natural, singkat, mudah diingat, dan cocok untuk audiens internasional."
        else:
            arahan_target = "Buat kombinasi judul berbahasa Indonesia dan Inggris; sebagian terasa lokal dan sebagian punya daya tarik global."
        prompt_judul = (
            f"Buat {jumlah_judul} pilihan judul lagu yang kuat, catchy, mudah diingat, dan tidak generik. "
            f"{tema_lengkap}. Genre: '{genre_str}'. Target: {target_pasar}. {arahan_target} "
            "Variasikan sudut emosi dan rasa penasaran. Hindari judul yang terlalu panjang atau berulang. "
            "Tampilkan daftar bernomor saja, tanpa penjelasan tambahan."
        )
        with st.spinner("Membuat ide judul sesuai target pendengar..."):
            hasil_judul = generate_ai_response(prompt_judul)
        st.session_state["generated_title_ideas"] = hasil_judul

if st.session_state.get("generated_title_ideas"):
    st.text_area("Hasil ide judul (bisa disalin):", value=st.session_state["generated_title_ideas"], height=180)

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
            prompt_lirik = f"Buatkan lirik lagu yang emosional, catchy, dan high-retention untuk Suno AI dengan Judul: '{judul_terpilih}', Tema: '{tema_lengkap}', Genre: '{genre_str}', target pendengar: '{target_pasar}'. Gunakan struktur tag meta Suno AI [Verse 1], [Chorus], [Bridge], [Outro]."
            lirik_res = generate_ai_response(prompt_lirik)
            st.session_state[lirik_state_key] = lirik_res

lirik_input = st.text_area("Lirik Lagu (Digunakan untuk Lirik Berjalan & Hook Otomatis):", value=st.session_state[lirik_state_key], height=220)

# ----------------------------------------------------
# 4. OPTIMIZER STYLE / PROMPT SUNO
# ----------------------------------------------------
st.divider()
st.subheader("🎛️ 4. Suno AI Style Prompt Optimizer")

if st.button("⚡ Buat Style Prompt Suno (Max 120 Karakter)"):
    prompt_style = f"Buatkan ringkasan Style / Prompt Musik untuk kolom Style of Music di Suno AI berdasarkan genre '{genre_str}', tema '{tema_lengkap}', dan target pendengar '{target_pasar}'. Karakter maksimal 120 karakter, dalam bahasa Inggris, dipisahkan koma."
    style_res = generate_ai_response(prompt_style)
    st.code(style_res, language="text")

# ----------------------------------------------------
# 5. VISUAL PROMPT GENERATOR (MIDJOURNEY / LEONARDO)
# ----------------------------------------------------
st.divider()
st.subheader("🖼️ 5. Generator Prompt Gambar Background")

if st.button("🎨 Buat Visual Prompt Artwork"):
    prompt_img = f"Buatkan 3 prompt gambar terperinci berbeda dalam bahasa Inggris untuk Midjourney / Leonardo AI yang menggambarkan 3 adegan berbeda dari lagu berjudul '{judul_terpilih}' dengan tema '{tema_lengkap}', nuansa genre '{genre_str}', dan target audiens '{target_pasar}'."
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

st.write("LONG: artwork sinematik + zoom perlahan + lirik berjalan + spektrum opsional + logo berdenyut. SHORT: 30 detik + pergantian artwork opsional + zoom perlahan + hook besar pada 4 detik pertama + CTA pada 4 detik terakhir.")

format_video = st.radio("Pilih Format Video:", ["📺 Long Video (16:9 Horizontal + Lirik Berjalan)", "📱 Short Video (9:16 Vertikal + Smart Hook 30 Detik)"], horizontal=True)

col_v1, col_v2, col_v3, col_v4 = st.columns(4)

with col_v1:
    uploaded_audio = st.file_uploader("1. Upload Audio (.mp3/.wav):", type=["mp3", "wav"])

with col_v2:
    uploaded_bg1 = st.file_uploader("2. Artwork Utama / 1:", type=["jpg", "jpeg", "png"])

with col_v3:
    uploaded_bg2 = st.file_uploader("3. Artwork 2 (Opsional — transisi otomatis):", type=["jpg", "jpeg", "png"])

with col_v4:
    uploaded_bg3 = st.file_uploader("4. Artwork 3 (Opsional — transisi otomatis):", type=["jpg", "jpeg", "png"])

st.caption("Untuk LONG maupun SHORT, unggah hingga 3 artwork. Jika hanya satu yang diunggah, gambar itu tetap digunakan sepanjang video.")

uploaded_logo = st.file_uploader("Upload Logo PNG Transparan (Opsional):", type=["png"])
uploaded_srt = st.file_uploader("Upload file lirik/subtitle .SRT (opsional — digunakan untuk render Long dan Short):", type=["srt"], help="Timestamp SRT menentukan kapan baris lirik tampil. Jika tidak diunggah, Long memakai pembagian lirik otomatis; Short tetap memakai hook dan CTA.")

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

effect_col1, effect_col2, effect_col3 = st.columns(3)
with effect_col1:
    effect_particles = st.checkbox("✨ Partikel bergerak", value=True)
with effect_col2:
    effect_glow = st.checkbox("🌟 Glow sinematik", value=True)
with effect_col3:
    effect_transitions = st.checkbox("🎞️ Transisi antar-artwork", value=True)

cta_text = ""
make_three_shorts = False
if "Short" in format_video:
    cta_text = st.text_input("Teks CTA (Muncul di Akhir Video Short):", value="Dengerin versi full-nya & Subscribe ya! 🎵")
    make_three_shorts = st.checkbox(
        "🔥 Mode 3 Shorts terpisah: 1 gambar = 1 video + hook berbeda",
        value=True,
        help="Jika 2 atau 3 gambar diunggah, setiap gambar akan dirender menjadi file Shorts sendiri. Tidak digabung menjadi satu video."
    )

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

                # Bersihkan lirik dan buang teks placeholder agar tidak masuk ke video.
                placeholder_phrases = (
                    "tulis atau generate lirik", "tulis lirik di sini",
                    "generate lirik di sini", "masukkan lirik", "lirik lagu di sini",
                )
                raw_lines = lirik_input.split('\n')
                clean_lyrics = []
                for line in raw_lines:
                    line_str = line.strip()
                    if not line_str or line_str.startswith('['):
                        continue
                    if any(phrase in line_str.lower() for phrase in placeholder_phrases):
                        continue
                    clean_lyrics.append(line_str)

                # Hook otomatis dari lirik; jika lirik belum ada, pakai judul dan bukan placeholder.
                if clean_lyrics:
                    chorus_idx = len(clean_lyrics) // 2 if len(clean_lyrics) > 2 else 0
                    auto_hook_text = clean_lyrics[chorus_idx]
                else:
                    auto_hook_text = judul_terpilih.strip() or "DENGARKAN BAGIAN TERKUAT"

                # Subtitle SRT opsional. Waktu baris mengikuti timestamp SRT;
                # kata di dalam setiap baris dibagi rata sebagai efek karaoke.
                timed_lyrics = []
                if uploaded_srt is not None:
                    try:
                        srt_text = uploaded_srt.getvalue().decode("utf-8-sig", errors="replace")
                        timestamp_pattern = re.compile(
                            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
                            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})"
                        )
                        def srt_seconds(stamp):
                            hh, mm, rest = stamp.replace(",", ".").split(":")
                            return int(hh) * 3600 + int(mm) * 60 + float(rest)
                        chunks = re.split(r"\n\s*\n", srt_text.strip())
                        for chunk in chunks:
                            match = timestamp_pattern.search(chunk)
                            if not match:
                                continue
                            after_stamp = chunk[match.end():]
                            cue_text = " ".join(
                                line.strip() for line in after_stamp.splitlines()
                                if line.strip() and not line.strip().isdigit()
                            )
                            cue_text = re.sub(r"<[^>]+>", "", cue_text).strip()
                            if cue_text:
                                timed_lyrics.append((
                                    srt_seconds(match.group(1)),
                                    srt_seconds(match.group(2)),
                                    cue_text
                                ))
                    except Exception:
                        timed_lyrics = []

                # Siapkan tiga hook yang berbeda untuk mode Shorts terpisah.
                auto_hooks = [auto_hook_text]
                if is_short:
                    fallback_hooks = []
                    if clean_lyrics:
                        pick_indices = [0, len(clean_lyrics) // 2, len(clean_lyrics) - 1]
                        for idx in pick_indices:
                            candidate = clean_lyrics[max(0, min(idx, len(clean_lyrics) - 1))].strip()
                            if candidate and candidate not in fallback_hooks:
                                fallback_hooks.append(candidate)
                    # Pastikan fallback tetap berbeda dan ringkas.
                    fallback_hooks = [re.sub(r"\s+", " ", h)[:85].strip(" .,!?") for h in fallback_hooks]
                    defaults = [
                        (judul_terpilih.strip() or "DENGARKAN BAGIAN TERKUAT"),
                        "INI BUKAN SEKADAR LAGU BIASA",
                        "DENGARKAN SAMPAI BAGIAN TERAKHIR",
                    ]
                    auto_hooks = []
                    for i in range(3):
                        candidate = fallback_hooks[i] if i < len(fallback_hooks) else defaults[i]
                        if candidate in auto_hooks:
                            candidate = defaults[i]
                        auto_hooks.append(candidate)

                    if clean_lyrics and api_key:
                        try:
                            hook_prompt = (
                                "Buat TEPAT 3 hook pembuka yang sangat kuat, nendang, berbeda satu sama lain "
                                "untuk 3 YouTube Shorts dari lagu yang sama. Variasikan sudut: emosional, provokatif, "
                                "dan menohok/penasaran. Bahasa Indonesia, maksimal 8 kata per hook. "
                                "Jangan mengarang tema di luar lirik. Balas hanya 3 baris bernomor 1., 2., 3. "
                                "tanpa penjelasan.\n"
                                f"Judul: {judul_terpilih}\nLirik: " + " / ".join(clean_lyrics[:30])
                            )
                            ai_response = generate_ai_response(hook_prompt)
                            parsed_hooks = []
                            for line in ai_response.splitlines():
                                line = re.sub(r"^\s*(?:[1-3][.)-]?|[-*])\s*", "", line).strip().strip('\"')
                                if line and not line.startswith("❌") and len(line) <= 100:
                                    parsed_hooks.append(line)
                            if len(parsed_hooks) >= 3:
                                auto_hooks = parsed_hooks[:3]
                        except Exception:
                            pass
                    auto_hook_text = auto_hooks[0]

                bg_images = []
                available_bg_images = []
                img_raw_1 = Image.open(uploaded_bg1).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                bg_images.append(img_raw_1)
                available_bg_images.append(img_raw_1)

                if uploaded_bg2:
                    img_raw_2 = Image.open(uploaded_bg2).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                    bg_images.append(img_raw_2)
                    available_bg_images.append(img_raw_2)
                else:
                    bg_images.append(img_raw_1)

                if uploaded_bg3:
                    img_raw_3 = Image.open(uploaded_bg3).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                    bg_images.append(img_raw_3)
                    available_bg_images.append(img_raw_3)
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

                    # Transisi artwork otomatis untuk LONG dan SHORT.
                    segment_len = duration / max(1, len(bg_images))
                    segment_idx = min(int(t / max(segment_len, 0.001)), len(bg_images) - 1)
                    active_bg = bg_images[segment_idx]

                    scale = 1.0 + (0.10 * progress)
                    crop_w = max(1, int(target_w / scale))
                    crop_h = max(1, int(target_h / scale))
                    left = max(0, (target_w - crop_w) // 2)
                    top = max(0, (target_h - crop_h) // 2)
                    frame_bg = active_bg.crop((left, top, left + crop_w, top + crop_h)).resize(
                        (target_w, target_h), Image.Resampling.LANCZOS
                    )

                    if effect_transitions and len(bg_images) > 1:
                        local_t = t - segment_idx * segment_len
                        fade_len = min(1.0, segment_len * 0.12)
                        if segment_idx > 0 and local_t < fade_len:
                            prev_bg = bg_images[segment_idx - 1]
                            prev_crop = prev_bg.crop((left, top, left + crop_w, top + crop_h)).resize(
                                (target_w, target_h), Image.Resampling.LANCZOS
                            )
                            alpha = min(1.0, max(0.0, local_t / max(fade_len, 0.001)))
                            frame_bg = Image.blend(prev_crop, frame_bg, alpha)

                    canvas = frame_bg.convert("RGB")
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

                    # Glow sinematik ringan (bloom) pada highlight.
                    if effect_glow:
                        glow_source = ImageEnhance.Brightness(canvas).enhance(1.35)
                        glow_source = glow_source.filter(ImageFilter.GaussianBlur(radius=8 if is_short else 6))
                        canvas = Image.blend(canvas.convert("RGB"), glow_source.convert("RGB"), 0.16)
                        draw = ImageDraw.Draw(canvas)

                    # Partikel bercahaya bergerak; posisi stabil antar-frame.
                    if effect_particles:
                        particle_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                        pd = ImageDraw.Draw(particle_layer)
                        particle_count = 34 if is_short else 26
                        for pi in range(particle_count):
                            seed = (pi * 7919) % 997
                            px = int((seed * 37 + t * (18 + (pi % 7) * 5)) % target_w)
                            py = int(((seed * 53) - t * (12 + (pi % 5) * 4)) % target_h)
                            radius = 2 + (pi % 3)
                            alpha = 85 + (pi % 5) * 25
                            pd.ellipse([px-radius, py-radius, px+radius, py+radius],
                                       fill=primary_color + (alpha,))
                        soft_particles = particle_layer.filter(ImageFilter.GaussianBlur(radius=5))
                        canvas = Image.alpha_composite(canvas.convert("RGBA"), soft_particles)
                        canvas = Image.alpha_composite(canvas, particle_layer).convert("RGB")
                        draw = ImageDraw.Draw(canvas)

                    if logo_img:
                        avg_bass = np.mean(norm_amps[:num_bars//4])
                        pulse_scale = 1.0 + (0.15 * avg_bass)
                        lw, lh = logo_img.size
                        new_lw, new_lh = int(lw * pulse_scale), int(lh * pulse_scale)
                        pulsed_logo = logo_img.resize((new_lw, new_lh), Image.Resampling.LANCZOS)
                        canvas.paste(pulsed_logo, (40, 40), pulsed_logo)

                    # Gunakan font TrueType agar teks tidak sekecil font default.
                    def load_font(size, bold=True):
                        font_paths = (
                            [
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                            ] if bold else [
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                            ]
                        )
                        for font_path in font_paths:
                            try:
                                return ImageFont.truetype(font_path, size=size)
                            except (OSError, IOError):
                                pass
                        return ImageFont.load_default()

                    def wrap_text(text, font, max_width):
                        words = str(text).split()
                        lines, current = [], ""
                        for word in words:
                            trial = word if not current else current + " " + word
                            if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
                                current = trial
                            else:
                                if current:
                                    lines.append(current)
                                current = word
                        if current:
                            lines.append(current)
                        return lines or [str(text)]

                    if not is_short:
                        if clean_lyrics:
                            # Karaoke approximation: baris dibagi rata sepanjang lagu,
                            # lalu kata aktif disorot berdasarkan waktu dalam baris.
                            if timed_lyrics:
                                cue = next(
                                    (item for item in timed_lyrics if item[0] <= t <= item[1]),
                                    None
                                )
                                if cue is None:
                                    active_lyric = ""
                                    words = []
                                    active_word_idx = -1
                                else:
                                    cue_start, cue_end, active_lyric = cue
                                    words = active_lyric.split()
                                    local_progress = (t - cue_start) / max(cue_end - cue_start, 0.001)
                                    active_word_idx = min(
                                        int(local_progress * max(1, len(words))), max(0, len(words) - 1)
                                    )
                            else:
                                lyric_slot = duration / max(1, len(clean_lyrics))
                                lyric_index = min(int(t / max(lyric_slot, 0.001)), len(clean_lyrics) - 1)
                                active_lyric = clean_lyrics[lyric_index]
                                words = active_lyric.split()
                                local_progress = (t - lyric_index * lyric_slot) / max(lyric_slot, 0.001)
                                active_word_idx = min(
                                    int(local_progress * max(1, len(words))), max(0, len(words) - 1)
                                )
                            lyric_font = load_font(30)
                            if not active_lyric:
                                return np.asarray(canvas.convert("RGB"))
                            box_y = 545
                            panel = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                            pd = ImageDraw.Draw(panel)
                            pd.rounded_rectangle(
                                [55, box_y, target_w - 55, box_y + 112],
                                radius=22, fill=(0, 0, 0, 125),
                                outline=primary_color + (170,), width=2
                            )
                            canvas = Image.alpha_composite(canvas.convert("RGBA"), panel).convert("RGB")
                            draw = ImageDraw.Draw(canvas)
                            lyric_lines = wrap_text(active_lyric, lyric_font, target_w - 180)[:2]
                            line_y = box_y + 20
                            consumed = 0
                            for lyric_line in lyric_lines:
                                line_words = lyric_line.split()
                                widths = [draw.textbbox((0, 0), w, font=lyric_font)[2] for w in line_words]
                                space_w = draw.textbbox((0, 0), " ", font=lyric_font)[2]
                                total_w = sum(widths) + space_w * max(0, len(line_words) - 1)
                                x = (target_w - total_w) // 2
                                for word in line_words:
                                    active = consumed == active_word_idx
                                    color = (255, 225, 70) if active else (255, 255, 255)
                                    draw.text((x, line_y), word, fill=color, font=lyric_font,
                                              stroke_width=1, stroke_fill=(0, 0, 0))
                                    x += draw.textbbox((0, 0), word, font=lyric_font)[2] + space_w
                                    consumed += 1
                                line_y += 42
                    else:
                        if t <= 4.0 and auto_hook_text:
                            # Hook besar, maksimal dua baris; panel transparan mengikuti ukuran teks.
                            hook_font = load_font(72)
                            hook_lines = wrap_text(auto_hook_text, hook_font, target_w - 190)[:2]
                            hook_text = "\n".join(hook_lines)
                            text_bbox = draw.multiline_textbbox((0, 0), hook_text, font=hook_font,
                                                                spacing=10, stroke_width=2)
                            text_w = text_bbox[2] - text_bbox[0]
                            text_h = text_bbox[3] - text_bbox[1]
                            pad_x, pad_y = 34, 24
                            box_w = min(target_w - 70, text_w + pad_x * 2)
                            box_h = text_h + pad_y * 2
                            box_x = (target_w - box_w) // 2
                            box_y = 165
                            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                            overlay_draw = ImageDraw.Draw(overlay)
                            overlay_draw.rounded_rectangle(
                                [box_x, box_y, box_x + box_w, box_y + box_h],
                                radius=26, fill=(0, 0, 0, 142),
                                outline=primary_color + (235,), width=3,
                            )
                            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
                            draw = ImageDraw.Draw(canvas)
                            text_x = (target_w - text_w) // 2
                            text_y = box_y + pad_y - text_bbox[1]
                            draw.multiline_text(
                                (text_x, text_y), hook_text, fill=(255, 255, 255), font=hook_font,
                                spacing=10, stroke_width=2, stroke_fill=(0, 0, 0)
                            )

                        if t >= (duration - 4.0) and cta_text:
                            cta_font = load_font(46)
                            cta_lines = wrap_text(cta_text, cta_font, target_w - 190)[:2]
                            cta_text_wrapped = "\n".join(cta_lines)
                            cta_bbox = draw.multiline_textbbox((0, 0), cta_text_wrapped, font=cta_font,
                                                               spacing=8, stroke_width=2)
                            cta_w = cta_bbox[2] - cta_bbox[0]
                            cta_h = cta_bbox[3] - cta_bbox[1]
                            pad_x, pad_y = 28, 20
                            box_w = min(target_w - 70, cta_w + pad_x * 2)
                            box_h = cta_h + pad_y * 2
                            box_x = (target_w - box_w) // 2
                            box_y = target_h - box_h - 135
                            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                            overlay_draw = ImageDraw.Draw(overlay)
                            overlay_draw.rounded_rectangle(
                                [box_x, box_y, box_x + box_w, box_y + box_h],
                                radius=24, fill=(0, 0, 0, 142),
                                outline=accent_color + (230,), width=3,
                            )
                            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
                            draw = ImageDraw.Draw(canvas)
                            draw.multiline_text(
                                ((target_w - cta_w) // 2, box_y + pad_y - cta_bbox[1]),
                                cta_text_wrapped, fill=(255, 230, 80), font=cta_font, spacing=8,
                                stroke_width=2, stroke_fill=(0, 0, 0)
                            )

                    return np.asarray(canvas.convert("RGB"))

                # Render Shorts secara terpisah: satu artwork menghasilkan satu MP4.
                # Mode Long tetap menghasilkan satu video seperti sebelumnya.
                if is_short and make_three_shorts:
                    render_jobs = [
                        (image, auto_hooks[min(i, len(auto_hooks) - 1)], f"short_{i+1:02d}.mp4")
                        for i, image in enumerate(available_bg_images)
                    ]
                else:
                    render_jobs = [(None, auto_hook_text, "short_video.mp4" if is_short else "long_video.mp4")]

                rendered_files = []
                for job_index, (job_image, job_hook, output_name) in enumerate(render_jobs, start=1):
                    if job_image is not None:
                        bg_images = [job_image]
                    auto_hook_text = job_hook
                    progress_label = f"Merender {output_name} ({job_index}/{len(render_jobs)})..."
                    st.write(f"🎬 {progress_label} — Hook: **{job_hook if is_short else 'tidak digunakan'}**")
                    video_clip = VideoClip(make_frame, duration=duration)
                    try:
                        video_clip = video_clip.set_audio(audio_clip)
                    except AttributeError:
                        video_clip = video_clip.with_audio(audio_clip)

                    video_clip.write_videofile(
                        output_name,
                        fps=fps,
                        codec="libx264",
                        audio_codec="aac",
                        preset="ultrafast",
                        threads=2,
                        logger=None
                    )

                    with open(output_name, "rb") as video_file:
                        video_bytes = video_file.read()
                    rendered_files.append((output_name, video_bytes))
                    st.video(video_bytes)
                    st.download_button(
                        label=f"⬇️ Download {output_name}",
                        data=video_bytes,
                        file_name=output_name,
                        mime="video/mp4",
                        key=f"download_render_{output_name}"
                    )
                    try:
                        video_clip.close()
                    except Exception:
                        pass

                st.success(f"✅ Berhasil merender {len(rendered_files)} video terpisah!")
                try:
                    audio_clip.close()
                except Exception:
                    pass
                if audio_clip_full is not audio_clip:
                    try:
                        audio_clip_full.close()
                    except Exception:
                        pass

            except Exception as e:
                st.error(f"❌ Gagal merender video: {type(e).__name__}: {e}")


# ----------------------------------------------------
# 8. MERGER VIDEO (GABUNGKAN BEBERAPA VIDEO MP4)
# ----------------------------------------------------
st.divider()
st.subheader("🎞️ 8. Merger Video (Gabungkan MP4)")
st.write(
    "Unggah dua atau lebih video MP4. Video akan digabungkan sesuai urutan file "
    "yang kamu pilih, lalu hasilnya bisa diputar dan diunduh."
)

uploaded_merge_videos = st.file_uploader(
    "Pilih video yang akan digabungkan (MP4):",
    type=["mp4", "mov", "m4v"],
    accept_multiple_files=True,
    key="merge_video_uploader",
)

merge_output_name = st.text_input(
    "Nama file hasil merger:",
    value="video_gabungan.mp4",
    key="merge_output_name",
)

if st.button("🎬 Gabungkan Video Sekarang", type="primary", key="merge_videos_button"):
    if not uploaded_merge_videos or len(uploaded_merge_videos) < 2:
        st.warning("Unggah minimal 2 video untuk digabungkan.")
    else:
        merge_clips = []
        final_merge_clip = None
        temp_paths = []
        try:
            with st.spinner("Menggabungkan video... Harap tunggu sampai selesai."):
                for index, uploaded_video in enumerate(uploaded_merge_videos, start=1):
                    safe_ext = os.path.splitext(uploaded_video.name)[1].lower()
                    if safe_ext not in [".mp4", ".mov", ".m4v"]:
                        safe_ext = ".mp4"
                    temp_path = f"temp_merge_{index}{safe_ext}"
                    with open(temp_path, "wb") as temp_file:
                        temp_file.write(uploaded_video.getbuffer())
                    temp_paths.append(temp_path)
                    clip = VideoFileClip(temp_path)
                    merge_clips.append(clip)

                # MoviePy menggabungkan secara berurutan; method ini tersedia pada
                # MoviePy 1.x dan 2.x untuk komposisi klip standar.
                final_merge_clip = concatenate_videoclips(merge_clips, method="compose")

                output_name = os.path.basename(merge_output_name.strip() or "video_gabungan.mp4")
                if not output_name.lower().endswith(".mp4"):
                    output_name += ".mp4"

                final_merge_clip.write_videofile(
                    output_name,
                    codec="libx264",
                    audio_codec="aac",
                    fps=24,
                    preset="ultrafast",
                    threads=2,
                    logger=None,
                )

            with open(output_name, "rb") as result_file:
                merged_bytes = result_file.read()

            st.success("✅ Video berhasil digabungkan!")
            st.video(merged_bytes)
            st.download_button(
                label="⬇️ Download Video Gabungan",
                data=merged_bytes,
                file_name=output_name,
                mime="video/mp4",
                key="download_merged_video",
            )
        except Exception as e:
            st.error(f"❌ Gagal menggabungkan video: {type(e).__name__}: {e}")
        finally:
            if final_merge_clip is not None:
                try:
                    final_merge_clip.close()
                except Exception:
                    pass
            for clip in merge_clips:
                try:
                    clip.close()
                except Exception:
                    pass
            for temp_path in temp_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass

