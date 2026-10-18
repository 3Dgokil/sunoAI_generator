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
    from moviepy.editor import AudioFileClip, VideoClip, concatenate_videoclips, concatenate_audioclips, VideoFileClip
except ImportError:
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.VideoClip import VideoClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    from moviepy.audio.AudioClip import concatenate_audioclips
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
# MANAGEMENT CLARIOHUB API
# Menggunakan OpenAI-compatible API ClarioHub, sama seperti Cadas Resonance.
# Simpan key di Streamlit Secrets:
# CLARIOHUB_API_KEY = "API_KEY_MILIKMU"
# ----------------------------------------------------
from openai import OpenAI

BASE_URL = "https://clariohub.id/v1"

st.sidebar.header("🔑 Status ClarioHub API")

try:
    api_key = str(st.secrets.get("CLARIOHUB_API_KEY", "")).strip()
except Exception:
    api_key = ""

if not api_key:
    api_key = os.getenv("CLARIOHUB_API_KEY", "").strip()

if api_key:
    st.sidebar.success("ClarioHub API Key terbaca dari Secrets")
else:
    st.sidebar.error(
        "API Key belum ditemukan. Tambahkan CLARIOHUB_API_KEY di "
        "Streamlit Cloud > App settings > Secrets."
    )

MODEL_OPTIONS = {
    "GPT-5.6 Sol (reasoning kuat)": "clario/gpt-5.6-sol",
    "DeepSeek V4 Pro": "clario/deepseek-v4-pro",
    "Gemini 3.7 Flash (cepat)": "clario/gemini-3.7-flash",
    "DeepSeek V4 Flash (hemat/cepat)": "clario/deepseek-v4-flash",
}

st.sidebar.subheader("🎯 Model per Tugas")
st.sidebar.caption("Model memakai API key ClarioHub yang sama. GPT-5.6 Sol membutuhkan token output lebih besar dan bisa lebih lambat.")

# Default tugas kreatif memakai GPT-5.6 Sol jika tersedia pada key ClarioHub.
lyrics_model_label = st.sidebar.selectbox(
    "✍️ Model untuk LIRIK", list(MODEL_OPTIONS.keys()), index=0, key="lyrics_model_label"
)
style_model_label = st.sidebar.selectbox(
    "🎛️ Model untuk STYLE OF MUSIC", list(MODEL_OPTIONS.keys()), index=0, key="style_model_label"
)
artwork_model_label = st.sidebar.selectbox(
    "🖼️ Model untuk PROMPT ARTWORK", list(MODEL_OPTIONS.keys()), index=1, key="artwork_model_label"
)
selected_model = st.sidebar.selectbox(
    "🤖 Model untuk tugas lain (judul/SEO/hook/video)", list(MODEL_OPTIONS.keys()), index=1, key="general_model_label"
)
TASK_MODEL_IDS = {
    "lyrics": MODEL_OPTIONS[lyrics_model_label],
    "style": MODEL_OPTIONS[style_model_label],
    "artwork": MODEL_OPTIONS[artwork_model_label],
    "general": MODEL_OPTIONS[selected_model],
}
st.sidebar.caption("Model lirik: " + TASK_MODEL_IDS["lyrics"])
st.sidebar.caption("Model style: " + TASK_MODEL_IDS["style"])
st.sidebar.caption("Model artwork: " + TASK_MODEL_IDS["artwork"])


def generate_ai_response(prompt_text, task="general"):
    """Panggil ClarioHub dengan model khusus sesuai tugas kreatif."""
    if not api_key:
        return (
            "❌ API Key belum ditemukan. Periksa Streamlit Secrets dan "
            "pastikan nama variabelnya CLARIOHUB_API_KEY."
        )
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=90.0,
            max_retries=1,
        )
        model_id = TASK_MODEL_IDS.get(task, TASK_MODEL_IDS["general"])
        # Model reasoning seperti GPT-5.6 Sol perlu ruang output lebih besar.
        output_tokens = 8192 if ("gpt-5.6-sol" in model_id or "opus-5" in model_id) else 4096
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah asisten profesional untuk produksi musik "
                        "Suno AI, penulis judul dan lirik orisinal. Ikuti tema "
                        "dan instruksi pengguna dengan akurat. Jangan mengubah "
                        "tema cerita hanya karena genre musik yang dipilih."
                    ),
                },
                {"role": "user", "content": str(prompt_text)},
            ],
            temperature=0.8,
            max_tokens=output_tokens,
        )
        result_text = response.choices[0].message.content
        if result_text and result_text.strip():
            return result_text.strip()
        st.error("ClarioHub mengembalikan respons kosong.")
        return "❌ Respons AI kosong. Silakan coba lagi."
    except Exception as e:
        st.error(
            f"ClarioHub gagal (model {TASK_MODEL_IDS.get(task, TASK_MODEL_IDS['general'])}, tugas {task}). "
            f"Detail: {type(e).__name__}: {e}"
        )
        return (
            "❌ AI belum berhasil membuat respons. Periksa nama model, "
            "API key, saldo, dan kuota di dashboard ClarioHub."
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
            "Gothic", "Phonk", "Ambient", "Blues", "Country", "Sound Horeg"
        ],
        default=["Pop", "Dangdut Modern"],
        help="Pilih satu atau beberapa genre untuk menggabungkan gaya musik."
    )

# Subgenre Sound Horeg tersedia sebagai pilihan tambahan; genre lama tetap utuh.
SOUND_HOREG_SUBGENRES = [
    "Sound Horeg Dangdut Koplo",
    "Sound Horeg Breakbeat",
    "Sound Horeg EDM Bass",
    "Sound Horeg Jaranan Jawa",
    "Sound Horeg Metal Koplo",
    "Sound Horeg Funkot",
    "Sound Horeg Electro House",
    "Sound Horeg Tribal / Rampak Kendang",
    "Sound Horeg Slow Bass",
    "Sound Horeg Bass Drop Festival",
    "Sound Horeg Campursari Remix",
    "Sound Horeg Techno",
    "Sound Horeg DJ Remix Jawa Timur",
    "Sound Horeg Full Bass Festival",
    "Sound Horeg Breakbeat Koplo",
    "Sound Horeg Dangdut Remix Pantura",
]
selected_horeg_subgenres = []
if "Sound Horeg" in genre_lagu:
    selected_horeg_subgenres = st.multiselect(
        "🔊 Pilih Subgenre Sound Horeg (bisa lebih dari satu):",
        SOUND_HOREG_SUBGENRES,
        default=["Sound Horeg Dangdut Koplo"],
        key="sound_horeg_subgenres",
        help="Pilih satu atau beberapa karakter. AI akan menggabungkannya secara seimbang ke Style of Music."
    )

# Bentuk satu referensi genre yang konsisten untuk judul, lirik, dan Style of Music.
genre_parts = list(genre_lagu) if genre_lagu else ["Pop"]
genre_parts.extend(selected_horeg_subgenres)
genre_str = ", ".join(dict.fromkeys(genre_parts))
subgenre_horeg_str = ", ".join(selected_horeg_subgenres) if selected_horeg_subgenres else "Tidak dipilih"
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
        # Tentukan arah cerita dari TEMA, bukan dari genre musik.
        # Genre mengatur warna musikal; genre tidak boleh mengubah tema cerita menjadi mistis.
        tema_lower = f"{tema_lagu} {subtema_lagu}".lower()
        kategori_tema = "sesuai tema utama yang diberikan pengguna"
        arahan_kategori = (
            "Pahami sendiri inti tema dan buat judul yang spesifik terhadap cerita tersebut. "
            "Jangan menambahkan unsur cerita yang tidak diminta."
        )
        kategori_keywords = [
            (("mistis", "mistik", "horor", "hantu", "gaib", "arwah", "kuntilanak", "pocong", "legenda", "ritual", "keris", "pesugihan", "makhluk halus", "nusantara mistis"),
             "mistis/legenda", "Gunakan unsur gaib, misteri, legenda, atau ritual HANYA karena tema ini memang memintanya."),
            (("cinta", "asmara", "kekasih", "pacar", "romantis", "rindu", "patah hati", "putus cinta", "selingkuh", "pengkhianatan", "mantan", "perpisahan", "ujung rambut"),
             "cinta/romansa", "Fokus pada cinta, rindu, kesetiaan, patah hati, perpisahan, atau pengkhianatan sesuai subtema. DILARANG memakai judul ritual, hantu, darah, mantra, keris, atau unsur mistis kecuali disebut jelas dalam tema."),
            (("ayah", "bapak", "ibu", "mama", "bunda", "orang tua", "anak", "istri", "suami", "keluarga", "pengorbanan keluarga", "saudara"),
             "keluarga", "Fokus pada hubungan keluarga, kasih sayang, pengorbanan, penyesalan, doa, dan kerinduan. Jangan ubah menjadi horor atau mistis."),
            (("perjuangan", "kerja keras", "hidup", "kemiskinan", "gaji", "utang", "beban hidup", "harapan", "mimpi", "kesuksesan", "pekerja", "buruh", "nasib"),
             "perjuangan hidup", "Fokus pada perjuangan, tekanan hidup, pengorbanan, kegagalan, dan harapan. Jangan mengubahnya menjadi cerita mistis."),
            (("sahabat", "teman", "tetangga", "persahabatan", "pertemanan"),
             "persahabatan/sosial", "Fokus pada persahabatan, kepercayaan, konflik sosial, atau solidaritas sesuai tema."),
            (("lucu", "humor", "komedi", "kocak", "kartun", "jenaka", "parodi"),
             "humor/komedi", "Buat judul lucu, ringan, mudah diingat, dan sesuai situasi komedi."),
            (("kritik sosial", "politik", "korupsi", "pejabat", "ketidakadilan", "sindiran", "satir"),
             "satir/kritik sosial", "Fokus pada sindiran sosial, ironi, ketidakadilan, atau kritik yang cerdas."),
        ]
        for keywords, kategori, arahan in kategori_keywords:
            if any(keyword in tema_lower for keyword in keywords):
                kategori_tema = kategori
                arahan_kategori = arahan
                break

        prompt_judul = (
            f"TUGAS: buat tepat {jumlah_judul} ide judul lagu yang kuat, catchy, mudah diingat, orisinal, dan tidak generik. "
            f"{tema_lengkap}. KATEGORI TEMA TERDETEKSI: {kategori_tema}. "
            f"Genre musik pilihan: '{genre_str}'. Target: {target_pasar}. {arahan_target} "
            f"ARAH CERITA WAJIB: {arahan_kategori} "
            "ATURAN PRIORITAS: tema dan subtema menentukan isi/jalan cerita; genre hanya menentukan karakter musik dan suasana bunyi. "
            "Jangan menganggap semua lagu bertema mistis. Jangan memasukkan darah, ritual, mantra, keris, hantu, arwah, atau horor "
            "kecuali tema/subtema secara jelas meminta unsur tersebut. Jika tema tentang cinta, keluarga, perjuangan, atau humor, "
            "semua judul harus tetap berada di tema itu meskipun genre musiknya metal, dangdut, gothic, atau bernuansa sinematik. "
            "STANDAR KUALITAS JUDUL WAJIB: judul harus realistis, ideal untuk lagu populer, alami seperti ungkapan yang benar-benar dipakai orang, "
            "langsung dipahami sekali baca, mudah diucapkan, mudah diingat, dan punya hubungan jelas dengan pengalaman atau cerita manusia. "
            "Utamakan bahasa sehari-hari yang kuat; jangan membuat frasa puitis yang janggal, terlalu abstrak, atau terdengar seperti hasil acak. "
            "Jangan terpaku pada kata benda atau kata unik di tema/judul awal. Misalnya, bila tema berisi frasa 'Cinta di Ujung Rambut', "
            "jangan otomatis memakai kata rambut, ujung, simpul, helai, atau jalinan pada semua alternatif; pahami maksud emosionalnya dan buat judul "
            "yang wajar tentang cinta, rindu, perpisahan, kesetiaan, atau patah hati. Pakai kata dari judul awal hanya jika memang penting bagi cerita. "
            "Setiap alternatif harus punya gagasan berbeda, bukan sekadar mengganti satu kata, mengacak sinonim, atau mengulang pola kalimat yang sama. "
            "Hindari kata-kata sensasional atau dramatis seperti berdarah, ritual, mantra, dan belati kecuali benar-benar relevan dengan cerita yang diminta. "
            "Sebelum menampilkan hasil, periksa setiap judul: apakah artinya langsung jelas, terdengar alami saat diucapkan, realistis untuk pengalaman manusia, "
            "relevan dengan tema/subtema, dan tidak terasa dipaksakan? Ganti judul yang gagal memenuhi standar tersebut. "
            "Judul harus bervariasi dari sisi emosi dan sudut cerita, tidak terlalu panjang, tidak berulang. "
            "Tampilkan daftar bernomor saja, satu judul per baris, tanpa penjelasan tambahan."
        )
        with st.spinner("Membuat ide judul sesuai target pendengar..."):
            hasil_judul = generate_ai_response(prompt_judul)
        st.session_state["generated_title_ideas"] = hasil_judul

# Ambil daftar judul dari respons AI agar judul bisa dipilih langsung.
def _parse_title_ideas(raw_text):
    parsed = []
    for raw_line in str(raw_text or "").splitlines():
        line = re.sub(r"^\s*(?:\d+[.)、:-]?|[-*•])\s*", "", raw_line).strip()
        line = line.strip("\"'` ")
        if not line or line.startswith("❌"):
            continue
        if len(line) > 100 or line.lower().startswith(("berikut", "ini adalah", "judul lagu:")):
            continue
        if line not in parsed:
            parsed.append(line)
    return parsed

judul_awal = tema_lagu.strip() if tema_lagu.strip() else "Lagu Tanpa Judul"
ide_judul = _parse_title_ideas(st.session_state.get("generated_title_ideas", ""))
if ide_judul:
    st.text_area("Hasil ide judul (bisa disalin):", value=st.session_state["generated_title_ideas"], height=180)
    selected_title_idea = st.selectbox("Pilih judul dari hasil AI:", ide_judul, key="selected_title_idea")
    if st.button("✅ Gunakan Judul Terpilih", key="apply_selected_title"):
        st.session_state["suno_title_input"] = selected_title_idea

if "suno_title_input" not in st.session_state:
    st.session_state["suno_title_input"] = judul_awal
judul_terpilih = st.text_input(
    "TITLE — Judul Lagu Utama / untuk Suno:",
    key="suno_title_input",
    help="Otomatis mengikuti judul yang dipilih dari hasil AI. Om juga bisa mengeditnya langsung."
)

# ----------------------------------------------------
# 3. GENERATOR LIRIK HIGH-RETENTION (SUNO FORMAT)
# Judul, genre/style, dan lirik ditampilkan terpisah agar mudah disalin ke Suno.
# ----------------------------------------------------
st.divider()
st.subheader("✍️ 3. Generator Lirik Lagu — Siap Salin ke Suno")
st.caption("Judul, Style of Music, dan Lyrics dipisahkan. Kolom Lyrics hanya berisi lirik beserta tag struktur Suno.")

lirik_state_key = "generated_lirik_text"
if lirik_state_key not in st.session_state:
    st.session_state[lirik_state_key] = "[Verse 1]\nLirik akan muncul di sini setelah dibuat."
if "suno_style_text" not in st.session_state:
    st.session_state["suno_style_text"] = ""

# Judul utama sudah dipilih/diedit pada bagian Generator Judul; tampilkan ringkasannya di sini.
st.caption(f"TITLE untuk Suno: **{st.session_state.get('suno_title_input', judul_terpilih)}**")

# Genre display mengikuti pilihan genre/subgenre di bagian atas secara otomatis.
st.text_input("GENRE — Referensi genre pilihan:", value=genre_str, disabled=True)

if st.button("🔥 Buat Lirik Lagu Lengkap", type="primary"):
    if not tema_lagu.strip():
        st.warning("Masukkan tema lagu terlebih dahulu!")
    else:
        with st.spinner("Membuat lirik saja dalam format Suno..."):
            prompt_lirik = f"""Tulis LIRIK LAGU LENGKAP yang orisinal untuk Suno AI.
Judul: {st.session_state.get('suno_title_input', judul_terpilih)}
Tema: {tema_lengkap}
Genre: {genre_str}
Target pendengar: {target_pasar}

IDENTITAS MUSIK: ikuti genre/subgenre pilihan pengguna secara spesifik: {genre_str}. Genre mengatur karakter bunyi, bukan mengubah tema cerita. Jangan otomatis mencampur metal, dangdut, atau unsur mistis jika tidak dipilih.
Jika genre mencakup Sound Horeg, bayangkan energi sound-system Jawa Timur dengan groove bass dan kick yang tegas; subgenre menentukan warna ritme/instrumen. Lirik tetap harus alami dan cocok dinyanyikan, bukan menuliskan instruksi sound effect atau daftar instrumen.
STRUKTUR: mulai dengan hook vokal kuat (tanpa intro penjelasan), lalu [Verse 1], [Pre-Chorus] bila cocok, [Chorus], [Verse 2], [Chorus], [Bridge] sesuai emosi cerita, [Final Chorus], dan [Outro]. Hindari intro lirik yang panjang.
BAHASA: gunakan bahasa yang sesuai target pendengar dan lirik santun lintas generasi.

ATURAN OUTPUT WAJIB:
- Keluarkan HANYA lirik lagu dan tag bagian Suno seperti [Hook], [Verse 1], [Chorus], [Bridge], [Final Chorus], [Outro].
- JANGAN tulis pengantar, komentar, analisis, markdown, judul, genre, label 'Lyrics:', atau penjelasan di luar lirik.
- Jangan beri tanda ** untuk bold.
- Jangan mengulang tema/genre sebagai paragraf.
- Pastikan lirik lengkap, memiliki chorus yang mudah diingat, dan bukan placeholder."""
            lirik_res = generate_ai_response(prompt_lirik, task="lyrics")

            # Bersihkan pembuka/metadata yang kadang ikut dikembalikan model.
            if lirik_res and not lirik_res.strip().startswith("❌"):
                cleaned = lirik_res.replace("```text", "").replace("```", "").strip()
                lines = cleaned.splitlines()
                # Bila model masih memberi pengantar, mulai dari tag struktur Suno pertama.
                tag_re = re.compile(r"^\s*\[(Hook|Intro|Verse\s*\d*|Pre-Chorus|Chorus|Bridge|Final Chorus|Outro|Instrumental|Break|Refrain)[^]]*\]\s*$", re.I)
                first_tag = next((i for i, line in enumerate(lines) if tag_re.match(line.strip())), None)
                if first_tag is not None:
                    cleaned = "\n".join(lines[first_tag:]).strip()
                # Hapus baris metadata judul/genre yang mungkin terselip di output.
                cleaned_lines = []
                for line in cleaned.splitlines():
                    if re.match(r"^\s*(judul|title|genre|gaya musik|lyrics|lirik lagu)\s*:", line, re.I):
                        continue
                    cleaned_lines.append(line)
                cleaned = "\n".join(cleaned_lines).strip()
                st.session_state[lirik_state_key] = cleaned
                st.session_state["lyrics_only_editor"] = cleaned
            else:
                st.session_state[lirik_state_key] = lirik_res
                st.session_state["lyrics_only_editor"] = lirik_res

st.markdown("**LYRICS — Salin kolom ini saja ke bagian Lyrics di Suno:**")
lirik_input = st.text_area(
    "Lirik lagu (untuk Suno dan lirik berjalan video):",
    key="lyrics_only_editor",
    value=st.session_state[lirik_state_key],
    height=360,
    help="Kolom ini khusus lirik dan tag struktur. Judul serta genre tersedia di kolom terpisah di atas."
)
# Sinkronkan teks yang diedit pengguna dengan nilai yang dipakai mesin render.
st.session_state[lirik_state_key] = lirik_input

# ----------------------------------------------------
# 4. OPTIMIZER STYLE / PROMPT SUNO
# ----------------------------------------------------
st.divider()
st.subheader("🎛️ 4. Suno AI Style of Music")
st.caption("Susun Style sesuai judul, tema, lirik, karakter vokal, dan pengaturan Suno. Advanced Options di bawah adalah panduan untuk diatur manual di Suno; Streamlit tidak dapat mengubah tombol Suno secara langsung.")

# Preset Advanced Options mengikuti genre yang dipilih.
# Angka adalah titik awal eksperimen, bukan nilai resmi/terbaik universal dari Suno.
def get_suno_genre_preset(genres):
    g = " ".join(genres).lower()
    if any(x in g for x in ["lagu anak", "kids", "nursery"]):
        return {"weirdness": 25, "style_influence": 80, "exclude": "dark, aggressive metal, explicit lyrics"}
    if any(x in g for x in ["rohani", "kasidah", "religious", "qasidah"]):
        return {"weirdness": 20, "style_influence": 85, "exclude": "aggressive vocals, harsh distortion, unrelated dance genres"}
    if any(x in g for x in ["balada", "ballad", "sad", "acoustic"]):
        return {"weirdness": 25, "style_influence": 85, "exclude": "upbeat party mood, aggressive rap, excessive distortion"}
    if any(x in g for x in ["horeg", "dangdut", "koplo"]):
        return {"weirdness": 40, "style_influence": 85, "exclude": "weak bass, slow ambient-only arrangement, unrelated genres"}
    if any(x in g for x in ["metal", "rock", "punk", "gothic"]):
        return {"weirdness": 35, "style_influence": 80, "exclude": "cheerful bubblegum pop, weak guitars, unrelated genres"}
    if any(x in g for x in ["edm", "house", "phonk", "synthwave", "techno"]):
        return {"weirdness": 35, "style_influence": 80, "exclude": "acoustic-only arrangement, unrelated slow ballad"}
    if any(x in g for x in ["orchestral", "cinematic", "mistis", "ambient"]):
        return {"weirdness": 30, "style_influence": 85, "exclude": "unrelated club beat, overly cheerful arrangement"}
    return {"weirdness": 30, "style_influence": 75, "exclude": "unrelated genres, contradictory mood"}

# Terapkan preset otomatis hanya ketika pilihan genre berubah; pengguna tetap boleh mengeditnya.
current_genre_signature = genre_str
if st.session_state.get("suno_genre_signature") != current_genre_signature:
    _preset = get_suno_genre_preset(genre_parts)
    st.session_state["suno_weirdness"] = _preset["weirdness"]
    st.session_state["suno_style_influence"] = _preset["style_influence"]
    st.session_state["suno_exclude_styles"] = _preset["exclude"]
    st.session_state["suno_genre_signature"] = current_genre_signature

with st.expander("⚙️ Advanced Options Suno", expanded=True):
    st.caption(f"Preset otomatis untuk genre saat ini: {genre_str}. Nilai dapat kamu ubah manual sebelum menyalin pengaturan ke Suno.")
    st.markdown("**Pengaturan utama**")
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        vocal_gender = st.radio("Vocal Gender", ["Male", "Female"], index=0, horizontal=True, key="suno_vocal_gender")
        duration_mode = st.radio("Duration", ["Custom", "Auto"], index=1, horizontal=True, key="suno_duration_mode")
        max_mode = st.radio("Max Mode", ["Off", "On"], index=0, horizontal=True, key="suno_max_mode")
        exclude_styles = st.text_input("Exclude styles", placeholder="Contoh: female vocals, rap, cheerful pop", key="suno_exclude_styles")
    with adv_col2:
        weirdness = st.slider("Weirdness", min_value=0, max_value=100, value=st.session_state.get("suno_weirdness", 30), step=5, help="Lebih rendah = lebih aman/terarah; lebih tinggi = lebih eksperimental.", key="suno_weirdness")
        style_influence = st.slider("Style Influence", min_value=0, max_value=100, value=st.session_state.get("suno_style_influence", 75), step=5, help="Lebih tinggi = Suno diarahkan lebih kuat mengikuti Style of Music.", key="suno_style_influence")
        variety = st.select_slider("Variety", options=["Low", "Normal", "High"], value="Normal", key="suno_variety")
        personalize = st.radio("Personalize / My Taste", ["Off", "On"], index=0, horizontal=True, key="suno_personalize")

    st.markdown("**Ringkasan untuk diatur di Suno**")
    st.code(
        f"Vocal Gender: {vocal_gender}\nDuration: {duration_mode}\nMax Mode: {max_mode}\n"
        f"Weirdness: {weirdness}%\nStyle Influence: {style_influence}%\n"
        f"Variety: {variety}\nPersonalize / My Taste: {personalize}\n"
        f"Exclude styles: {exclude_styles or '(kosong / tidak ada)'}",
        language="text"
    )

if st.button("⚡ Buat Style of Music Suno"):
    # Bangun style berdasarkan judul DAN isi lirik, bukan hanya tema/genre.
    style_title = st.session_state.get("suno_title_input", judul_terpilih).strip() or judul_terpilih
    style_lyrics = st.session_state.get("lyrics_only_editor", st.session_state.get(lirik_state_key, "")).strip()
    if not style_lyrics or style_lyrics.startswith("[Verse 1]") and "Lirik akan muncul" in style_lyrics:
        style_lyrics = "Lirik belum dibuat. Gunakan judul dan tema sebagai dasar emosi cerita."
    # Batasi konteks lirik supaya prompt tetap ringan, tetapi cukup untuk membaca cerita dan emosi.
    style_lyrics_context = style_lyrics[:5000]
    prompt_style = f"""You are a senior record producer and Suno prompt specialist. Create ONE production-ready English Style of Music prompt for the exact song below. The result must be concrete, emotionally specific, musically coherent, and usable directly in Suno Custom Mode.

SONG TITLE: {style_title}
MAIN THEME / SUBTHEME: {tema_lengkap}
USER-SELECTED GENRES (respect every selected genre; do not replace them with unrelated genres): {genre_str}
TARGET: {target_pasar}
REQUIRED LEAD VOCAL GENDER: {vocal_gender}. Treat this as a hard requirement. For Male, explicitly request a mature masculine male lead voice (baritone or tenor chosen to fit the song), chest resonance, and consistent male lead throughout; for Female, explicitly request a clearly feminine lead voice. Do not use a duet or backing lead of the opposite gender.
EXCLUDE STYLES / TRAITS: {exclude_styles.strip() if exclude_styles.strip() else "none specified"}. Do not include or recommend these.
USER SETTINGS (context only; do not print numeric settings in the prompt): Weirdness {weirdness}%, Style Influence {style_influence}%, Variety {variety}, Personalize {personalize}, Duration {duration_mode}, Max Mode {max_mode}.

LYRICS AND STORY — identify the actual relationship, event, emotional point of view, and emotional progression. Do not invent a different story:
{style_lyrics_context}

PRODUCTION RULES:
1) Build the emotional center from the actual story: for grief/regret/longing, use aching melodic phrasing, vulnerable restrained verses, expressive sustained notes, and a cathartic chorus; for anger/hope/romance/mystery, adapt the delivery accordingly. Avoid generic “dark emotional” wording without musical details.
2) Preserve the chosen genre blend. When the selected genres include melodic heavy metal/hard rock, define melodic distorted guitars, expressive lead-guitar lines, and controlled powerful drums. When modern dangdut is selected, give kendang and deep bass a distinct, audible groove without turning the whole song into a generic dance track. When orchestral is selected, use cinematic strings/brass/choir-like layers to build scale while leaving space for the lead vocal. Use Nusantara gamelan, gong, bamboo flute, or mystical ambience only when compatible with the theme and selected genres; state their role instead of listing instruments without purpose.
3) Specify vocal tone and delivery, groove/tempo feel, instrument roles, production texture, and a clear energy arc. Start with an immediate vocal hook or brief instrumental pickup (no long intro); intimate/tense verse; memorable singable chorus; emotional or mystical bridge; grand final chorus; lingering outro. Do not force every section to be loud.
4) Keep one coherent lead-vocal identity throughout. No contradictory moods, no cluttered arrangement, no unrelated genres, no spoken explanation, and no lyrics.
5) If Sound Horeg is selected, make it audible and specific rather than merely naming it: powerful but controlled sub-bass, hard-hitting kick, loud outdoor sound-system energy, punchy percussion, and a clear dance groove.
SELECTED SOUND HOREG SUBGENRES — use these exact choices as the musical blueprint: {subgenre_horeg_str}.
Apply the selected subgenres together where compatible, but keep a clear hierarchy and avoid a chaotic pile-up. Use these mappings as guidance: Dangdut Koplo = lively kendang and syncopated koplo groove; Breakbeat = broken syncopated electronic drums; EDM Bass/Bass Drop = build-and-drop structure with thick bass; Jaranan Jawa = appropriate Javanese percussion accents; Metal Koplo = metal guitars with koplo rhythm; Funkot = fast, bouncy Indonesian dance rhythm; Electro House/Techno = steady electronic club pulse; Tribal/Rampak Kendang = layered rhythmic percussion; Slow Bass = spacious half-time groove; Campursari Remix = tasteful gamelan/campursari colors; DJ Remix Jawa Timur = East Java remix/breakbeat feel; Full Bass Festival = powerful outdoor festival bass drops. Do not introduce unselected subgenres unless musically necessary. Keep the song's story and vocal melody clear.

OUTPUT: ONLY one polished English Style of Music paragraph, maximum 950 characters. No heading, bullet list, markdown, quotation marks, song title, lyric quotation, or explanation. Prioritize concrete musical instructions over adjective piles."""
    with st.spinner("Menganalisis judul, tema, dan lirik untuk menyusun Style of Music..."):
        st.session_state["suno_style_text"] = generate_ai_response(prompt_style, task="style")
        st.session_state["suno_style_editor"] = st.session_state["suno_style_text"]

style_input = st.text_area(
    "STYLE OF MUSIC — salin ke kolom Style of Music di Suno:",
    key="suno_style_editor",
    value=st.session_state["suno_style_text"],
    height=130,
    help="Hanya deskripsi musik berbahasa Inggris, bukan lirik."
)
st.session_state["suno_style_text"] = style_input
st.info("Di Suno Custom Mode: masukkan judul ke Title, prompt musik ke Style of Music, dan lirik ke Lyrics. Pastikan Instrumental OFF jika ingin lagu dengan vokal. Atur Vocal Gender, Duration, Max Mode, Weirdness, Style Influence, Variety, Personalize, dan Exclude styles sesuai ringkasan Advanced Options di atas.")

# ----------------------------------------------------
# 5. VISUAL PROMPT GENERATOR (THUMBNAIL + LONG + 3 SHORTS)
# ----------------------------------------------------
st.divider()
st.subheader("🖼️ 5. Generator Prompt Artwork YouTube")
st.caption("Buat 5 prompt terpisah: 1 thumbnail, 1 artwork video Long, dan 3 artwork Shorts. Setiap prompt punya komposisi/adegan sendiri, tetapi tokoh, cerita, dan suasananya tetap konsisten dengan judul serta tema lagu.")

import streamlit.components.v1 as components

_ARTWORK_TYPES = [
    ("THUMBNAIL YOUTUBE", "16:9", "High-impact YouTube music thumbnail. Strong focal subject, readable visual hierarchy, emotionally compelling expression, cinematic contrast, clear negative space for adding title text later. Do not render any letters or text inside the generated image."),
    ("ARTWORK VIDEO LONG", "16:9", "Cinematic widescreen 16:9 background for the full-length music video. A different scene from the thumbnail, immersive environment, balanced composition, main subject clearly visible, enough visual detail for slow zoom and lyric overlays. No text or typography."),
    ("ARTWORK SHORT 1", "9:16", "Vertical 9:16 composition for Short 1. Show the song's strongest opening emotional moment as a distinct scene, with a clear subject and strong mobile-screen readability. Keep important faces and objects away from extreme edges. No text."),
    ("ARTWORK SHORT 2", "9:16", "Vertical 9:16 composition for Short 2. Show a different story beat or emotional angle from Short 1, not merely a crop or recolor. Maintain the same characters and visual identity. Keep the focal subject clear on a phone screen. No text."),
    ("ARTWORK SHORT 3", "9:16", "Vertical 9:16 composition for Short 3. Show a third distinct story beat, such as a turning point, emotional reveal, or resolution appropriate to the actual song theme. Do not repeat the other two compositions. Keep visual continuity. No text."),
]

if "artwork_prompt_items" not in st.session_state:
    st.session_state["artwork_prompt_items"] = []
if "artwork_prompt_labels" not in st.session_state:
    st.session_state["artwork_prompt_labels"] = []

if st.button("🎨 Buat 5 Prompt: Thumbnail + Long + 3 Shorts", key="make_artwork_prompts"):
    prompt_img = f"""Create exactly FIVE complete, detailed English image-generation prompts for the same song.
Song title: '{judul_terpilih}'.
Chosen theme/story: '{tema_lengkap}'.
Music genre and mood: '{genre_str}'.
Target audience/language: '{target_pasar}'.

ACCURACY RULES:
- First infer the actual subject, relationship, conflict, and emotional tone from the title and chosen theme. Each image must clearly fit that same song/story.
- Never invent a mystical, horror, ritual, supernatural, or folklore element unless the chosen theme/title explicitly calls for it.
- Use believable, coherent scenes and natural human expressions. Avoid random symbolic objects that do not support the story.
- Maintain continuity in recurring characters (age range, appearance, clothing, and environment) across all five images, but give every image a genuinely different scene, camera angle, framing, and story moment.
- Photorealistic, cinematic, polished professional music-artwork quality, rich but controlled lighting, sharp focal subject, realistic anatomy and hands.
- No text, letters, title typography, logos, watermarks, subtitles, UI, or borders inside any generated image. The thumbnail should leave clean negative space so title text can be added later in an editor.
- Follow the exact aspect ratio and purpose for each item below. Do not make all five images vertical. No explanations, advice, or tool-specific instructions.

OUTPUT FORMAT — use these exact labels and provide one standalone prompt of approximately 100–150 words under each:
[THUMBNAIL YOUTUBE | 16:9]
{_ARTWORK_TYPES[0][2]}
[ARTWORK VIDEO LONG | 16:9]
{_ARTWORK_TYPES[1][2]}
[ARTWORK SHORT 1 | 9:16]
{_ARTWORK_TYPES[2][2]}
[ARTWORK SHORT 2 | 9:16]
{_ARTWORK_TYPES[3][2]}
[ARTWORK SHORT 3 | 9:16]
{_ARTWORK_TYPES[4][2]}

Return only these five labeled prompts. Do not merge them together."""
    with st.spinner("Membuat 5 prompt dengan rasio dan adegan yang sesuai..."):
        img_res = generate_ai_response(prompt_img, task="artwork")

    import re as _re
    labels = [item[0] for item in _ARTWORK_TYPES]
    prompts = []
    pattern = r"(?is)\[(THUMBNAIL YOUTUBE\s*\|\s*16:9|ARTWORK VIDEO LONG\s*\|\s*16:9|ARTWORK SHORT [123]\s*\|\s*9:16)\]\s*"
    matches = list(_re.finditer(pattern, str(img_res)))
    if matches:
        parsed = {}
        for i, match in enumerate(matches):
            label = match.group(1).split("|")[0].strip().upper()
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(str(img_res))
            parsed[label] = str(img_res)[content_start:content_end].strip()
        for label in labels:
            text_value = parsed.get(label, "")
            text_value = _re.sub(r"^\s*(?:\(?(?:16:9|9:16)\)?\s*[-:—]?\s*)", "", text_value)
            prompts.append(text_value)

    if len(prompts) == 5 and all(len(x) > 40 for x in prompts):
        st.session_state["artwork_prompt_items"] = prompts
        st.session_state["artwork_prompt_labels"] = labels
    else:
        st.warning("AI belum memisahkan semua 5 prompt dengan benar. Coba tekan tombol sekali lagi. Respons mentah ditampilkan agar tidak salah menempatkan rasio gambar.")
        st.session_state["artwork_prompt_items"] = [str(img_res).strip()]
        st.session_state["artwork_prompt_labels"] = ["HASIL AI — PERLU DIPERIKSA"]

for _idx, _prompt_text in enumerate(st.session_state["artwork_prompt_items"]):
    _label = st.session_state["artwork_prompt_labels"][_idx] if _idx < len(st.session_state["artwork_prompt_labels"]) else f"ARTWORK {_idx + 1}"
    _ratio = next((ratio for label, ratio, _desc in _ARTWORK_TYPES if label == _label), "Periksa format")
    _safe_prompt_json = json.dumps(_prompt_text, ensure_ascii=False)
    _safe_label_json = json.dumps(_label, ensure_ascii=False)
    _safe_ratio_json = json.dumps(_ratio, ensure_ascii=False)
    _html = f'''<!doctype html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{{box-sizing:border-box}} body{{margin:0;padding:4px;font-family:Arial,sans-serif;background:transparent;color:#f5f5f7}}
.prompt-card{{border:1px solid #555965;border-radius:14px;background:#202127;padding:16px;cursor:pointer}}
.head{{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}}
.title{{font-size:21px;font-weight:800;line-height:1.3;flex:1;min-width:170px}}
.ratio{{display:inline-block;margin-top:5px;border-radius:6px;background:#343743;padding:4px 8px;font-size:14px;font-weight:700;color:#f4d34e}}
.copy{{border:0;border-radius:9px;padding:12px 15px;background:#f4d34e;color:#171717;font-weight:800;font-size:16px;cursor:pointer;white-space:nowrap}}
.body{{font-size:18px;line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere;user-select:text}}
.status{{font-size:14px;color:#a7f3c0;min-height:20px;margin-top:9px}}
</style></head><body>
<div class="prompt-card" id="card">
 <div class="head"><div class="title" id="title"></div><button class="copy" id="copyBtn">SALIN PROMPT</button></div>
 <div class="body" id="promptText"></div><div class="status" id="status"></div>
</div>
<script>
const promptText = {_safe_prompt_json};
document.getElementById('title').textContent = {_safe_label_json};
const ratio = {_safe_ratio_json};
const ratioEl = document.createElement('div'); ratioEl.className='ratio'; ratioEl.textContent='RASIO '+ratio;
document.getElementById('title').appendChild(document.createElement('br'));
document.getElementById('title').appendChild(ratioEl);
document.getElementById('promptText').textContent = promptText;
async function copyPrompt() {{
 try {{
  if (navigator.clipboard && navigator.clipboard.writeText) {{ await navigator.clipboard.writeText(promptText); }}
  else {{ const t=document.createElement('textarea'); t.value=promptText; t.style.position='fixed'; t.style.opacity='0'; document.body.appendChild(t); t.select(); const ok=document.execCommand('copy'); t.remove(); if(!ok) throw new Error('copy failed'); }}
  document.getElementById('status').textContent='✓ Tersalin! Buka tool gambar, lalu tempel prompt ini.';
  document.getElementById('copyBtn').textContent='TERSALIN ✓';
 }} catch(e) {{ document.getElementById('status').textContent='Clipboard diblokir browser. Tekan lama teks lalu pilih Salin.'; }}
}}
document.getElementById('copyBtn').addEventListener('click', function(e){{e.stopPropagation();copyPrompt();}});
document.getElementById('card').addEventListener('click', copyPrompt);
</script></body></html>'''
    components.html(_html, height=min(900, max(360, 205 + len(_prompt_text) // 2)), scrolling=True)

# ----------------------------------------------------
# 5B. VIDEO AI PROMPT GENERATOR (STATIC CINEMATIC / TRIPOD)
# ----------------------------------------------------
st.divider()
st.subheader("🎞️ 5B. Generator Prompt Video AI")
st.caption("Buat prompt video bergerak dari cerita lagu. Default kamera terkunci seperti contoh Gemini: no zoom, static cinematic, tripod mode. Kamera tetap diam; hanya elemen adegan yang diminta boleh bergerak.")

_video_motion_mode = st.selectbox(
    "Gaya gerakan video:",
    ["Static Cinematic / Tripod", "Subtle Motion / Tripod", "Dynamic Camera"],
    index=0,
    key="video_ai_motion_mode",
)
_video_duration = st.selectbox("Durasi tiap klip:", ["5 seconds", "8 seconds", "10 seconds"], index=2, key="video_ai_duration")
_video_subject_motion = st.text_input(
    "Gerakan khusus (opsional):",
    value="",
    placeholder="Contoh: asap hijau bergerak perlahan, hujan turun, rambut tertiup angin",
    key="video_ai_subject_motion",
)

_VIDEO_PROMPT_LABELS = [
    "VIDEO LONG — ADEGAN 1",
    "VIDEO LONG — ADEGAN 2",
    "VIDEO SHORT 1 — HOOK",
    "VIDEO SHORT 2 — CERITA",
    "VIDEO SHORT 3 — KLIMAKS",
]
if "video_ai_prompt_items" not in st.session_state:
    st.session_state["video_ai_prompt_items"] = []
if "video_ai_prompt_labels" not in st.session_state:
    st.session_state["video_ai_prompt_labels"] = []

if st.button("🎬 Buat 5 Prompt Video AI", key="make_video_ai_prompts"):
    if _video_motion_mode == "Static Cinematic / Tripod":
        camera_instruction = "Static cinematic shot, locked tripod camera. NO zoom in, NO zoom out, NO pan, NO tilt, NO camera shake, NO camera movement."
    elif _video_motion_mode == "Subtle Motion / Tripod":
        camera_instruction = "Locked tripod camera, no zoom, no pan, no tilt, no camera shake. Only very subtle natural movement in the scene."
    else:
        camera_instruction = "Cinematic camera movement may be used sparingly when it supports the scene; keep motion smooth, controlled, and physically plausible."

    custom_motion = _video_subject_motion.strip() or "Only animate scene elements that naturally fit the story, such as subtle breathing, hair or clothing moved by wind, rain, drifting mist, floating dust, or gentle light flicker. Do not add unrelated effects."
    prompt_video = f'''Create exactly FIVE separate English image-to-video prompts for a music video. Each prompt should be ready to copy into an AI video tool such as Gemini, Kling, or Runway.
SONG TITLE: {judul_terpilih}
THEME / STORY: {tema_lengkap}
GENRE / MOOD: {genre_str}
AUDIENCE / LANGUAGE: {target_pasar}
CLIP DURATION: {_video_duration}
CAMERA MODE: {_video_motion_mode}
USER-REQUESTED MOTION: {custom_motion}

CORE RULES FOR ALL FIVE PROMPTS:
- Understand the actual song theme and story. Do not introduce horror, mystical, ritual, folklore, or supernatural content unless the chosen theme/title calls for it.
- Treat the supplied image as the visual source. Preserve the original character identity, face, clothing, composition, framing, environment, colors, and art direction. Do not redesign the character.
- {camera_instruction}
- Camera lock does NOT mean a frozen image: animate only relevant subject/environment details with subtle believable motion. Keep the main subject stable and recognizable.
- No scene cuts, no new characters, no new objects, no morphing, no warped face or hands, no flickering, no sudden lighting changes, no text/logo changes, no subtitles, no watermark.
- Cinematic lighting, natural physical motion, stable background, smooth motion, consistent details for the entire {_video_duration} clip.
- If the image contains title text or a logo, preserve it exactly; do not generate new writing.
- Make each prompt concise but specific, approximately 70-110 words. Return only five labeled prompts, no introduction or advice.

Create these distinct clips:
[VIDEO LONG - ADEGAN 1]
A calm establishing shot that introduces the song's world and emotional atmosphere. Keep the main character and scene faithful to the input image.
[VIDEO LONG - ADEGAN 2]
A different emotional moment suitable later in the full-length song. Maintain character continuity, but describe a distinct natural subject/environment motion.
[VIDEO SHORT 1 - HOOK]
An immediately compelling first-second visual moment for a vertical Short. Keep the focal subject clear and mobile-screen friendly.
[VIDEO SHORT 2 - CERITA]
A different story beat and motion from Short 1, not just a repeat or crop.
[VIDEO SHORT 3 - KLIMAKS]
A stronger emotional turning point or resolution appropriate to the actual theme, without inventing unrelated events.

For every label, provide a standalone complete image-to-video prompt. Repeat the camera and preservation instructions inside each prompt so each can be copied independently.'''
    with st.spinner("Menyusun lima prompt video sesuai tema dan mode kamera..."):
        video_res = generate_ai_response(prompt_video)

    import re as _re_video
    video_pattern = r"(?is)\[(VIDEO LONG\s*[—-]\s*ADEGAN [12]|VIDEO SHORT [123]\s*[—-]\s*(?:HOOK|CERITA|KLIMAKS))\]\s*"
    video_matches = list(_re_video.finditer(video_pattern, str(video_res)))
    parsed_video = {}
    for _i, _match in enumerate(video_matches):
        _key = _re_video.sub(r"\s+", " ", _match.group(1).replace("—", "-")).strip().upper()
        _start = _match.end()
        _end = video_matches[_i + 1].start() if _i + 1 < len(video_matches) else len(str(video_res))
        parsed_video[_key] = str(video_res)[_start:_end].strip()
    expected_video_keys = [
        "VIDEO LONG - ADEGAN 1", "VIDEO LONG - ADEGAN 2",
        "VIDEO SHORT 1 - HOOK", "VIDEO SHORT 2 - CERITA", "VIDEO SHORT 3 - KLIMAKS"
    ]
    parsed_items = [parsed_video.get(_k, "") for _k in expected_video_keys]
    if all(len(_v) > 40 for _v in parsed_items):
        st.session_state["video_ai_prompt_items"] = parsed_items
        st.session_state["video_ai_prompt_labels"] = _VIDEO_PROMPT_LABELS
    else:
        st.warning("Respons AI belum berhasil dipisah menjadi lima prompt. Respons mentah ditampilkan agar bisa diperiksa.")
        st.session_state["video_ai_prompt_items"] = [str(video_res).strip()]
        st.session_state["video_ai_prompt_labels"] = ["HASIL AI — PERIKSA FORMAT"]

for _vidx, _video_prompt in enumerate(st.session_state["video_ai_prompt_items"]):
    _vlabel = st.session_state["video_ai_prompt_labels"][_vidx] if _vidx < len(st.session_state["video_ai_prompt_labels"]) else f"VIDEO PROMPT {_vidx + 1}"
    _vjson = json.dumps(_video_prompt, ensure_ascii=False)
    _vljson = json.dumps(_vlabel, ensure_ascii=False)
    _vhtml = f'''<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>
*{{box-sizing:border-box}}body{{margin:0;padding:4px;font-family:Arial,sans-serif;background:transparent;color:#f5f5f7}}.card{{border:1px solid #555965;border-radius:14px;background:#202127;padding:16px}}.head{{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}}.title{{font-size:20px;font-weight:800;line-height:1.3;flex:1;min-width:160px}}button{{border:0;border-radius:9px;padding:12px 15px;background:#f4d34e;color:#171717;font-weight:800;font-size:16px;cursor:pointer}}.body{{font-size:18px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere;user-select:text}}.status{{font-size:14px;color:#a7f3c0;min-height:20px;margin-top:9px}}
</style></head><body><div class="card" id="card"><div class="head"><div class="title" id="title"></div><button id="copyBtn">SALIN PROMPT VIDEO</button></div><div class="body" id="body"></div><div class="status" id="status"></div></div><script>
const promptText={_vjson};document.getElementById('title').textContent={_vljson};document.getElementById('body').textContent=promptText;
async function copyPrompt(){{try{{if(navigator.clipboard&&navigator.clipboard.writeText){{await navigator.clipboard.writeText(promptText)}}else{{const t=document.createElement('textarea');t.value=promptText;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();const ok=document.execCommand('copy');t.remove();if(!ok)throw new Error('copy failed')}}document.getElementById('status').textContent='✓ Tersalin! Tempel prompt ini di tool video AI.';document.getElementById('copyBtn').textContent='TERSALIN ✓'}}catch(e){{document.getElementById('status').textContent='Clipboard diblokir browser. Tekan lama teks lalu pilih Salin.'}}}}document.getElementById('copyBtn').addEventListener('click',copyPrompt);</script></body></html>'''
    components.html(_vhtml, height=min(850, max(320, 190 + len(_video_prompt) // 2)), scrolling=True)

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
# 6B. ALBUM RENDERER — MULTI AUDIO + MULTI ARTWORK
# ----------------------------------------------------
st.divider()
st.subheader("🎞️ 6B. Album Renderer — Multi Audio + Multi Artwork")
st.caption("Mode terpisah: gabungkan banyak lagu dan gunakan banyak visual MP4/JPG/PNG secara bergantian. Renderer Long/Short lama tetap dipertahankan.")

album_audio_files = st.file_uploader(
    "🎵 Upload banyak audio album (.mp3/.wav):", type=["mp3", "wav"],
    accept_multiple_files=True, key="album_audio_files",
    help="Jumlah file tidak dibatasi oleh kode; batas nyata mengikuti kapasitas server dan batas upload Streamlit."
)
album_visual_files = st.file_uploader(
    "🎬 Upload banyak visual (.mp4/.jpg/.jpeg/.png):", type=["mp4", "jpg", "jpeg", "png"],
    accept_multiple_files=True, key="album_visual_files",
    help="Visual diputar bergantian sesuai urutan upload, lalu diulang bila durasi album lebih panjang."
)
album_resolution = st.selectbox("Resolusi Album", ["720p — lebih ringan", "1080p — lebih tajam"], index=0, key="album_resolution")
album_fps = st.selectbox("FPS Album", [20, 24, 30], index=1, key="album_fps")
album_show_lyrics = st.checkbox("📝 Lirik berjalan", value=True, key="album_show_lyrics")
album_show_spectrum = st.checkbox("🎚️ Spectrum audio", value=True, key="album_show_spectrum")
album_lirik_text = st.text_area("Lirik untuk album (opsional; satu baris per baris lirik)", value="", height=110, key="album_lirik_text")
st.caption("Preset ringan: visual MP4 + lirik berjalan + spectrum. Tanpa zoom, glow, atau partikel tambahan.")

if st.button("🎬 Render Album Multi-File", type="primary", key="render_album_multi"):
    if not album_audio_files:
        st.warning("Unggah minimal satu audio terlebih dahulu.")
    elif not album_visual_files:
        st.warning("Unggah minimal satu visual terlebih dahulu.")
    else:
        audio_clips_album, visual_clips_album, temp_paths_album = [], [], []
        combined_audio_album = final_clip_album = None
        try:
            with st.spinner("Menggabungkan lagu dan merender album..."):
                for i, af in enumerate(album_audio_files, 1):
                    ext = os.path.splitext(af.name)[1].lower() or ".mp3"
                    path = f"album_audio_{i}{ext}"
                    with open(path, "wb") as f: f.write(af.getbuffer())
                    temp_paths_album.append(path)
                    audio_clips_album.append(AudioFileClip(path))
                combined_audio_album = concatenate_audioclips(audio_clips_album)
                total_duration = float(combined_audio_album.duration)
                if total_duration <= 0: raise ValueError("Durasi audio gabungan tidak valid.")

                for i, vf in enumerate(album_visual_files, 1):
                    ext = os.path.splitext(vf.name)[1].lower()
                    if ext == ".mp4":
                        path = f"album_visual_{i}.mp4"
                        with open(path, "wb") as f: f.write(vf.getbuffer())
                        temp_paths_album.append(path)
                        clip = VideoFileClip(path, audio=False)
                        if not clip.duration or clip.duration <= 0:
                            clip.close(); raise ValueError(f"Durasi visual {vf.name} tidak valid.")
                        visual_clips_album.append(("video", clip, float(clip.duration)))
                    else:
                        visual_clips_album.append(("image", Image.open(vf).convert("RGB"), 12.0))

                w, h = (1920, 1080) if album_resolution.startswith("1080p") else (1280, 720)
                fps = int(album_fps)
                lyrics = [re.sub(r"\s+", " ", x).strip() for x in album_lirik_text.splitlines() if x.strip() and not x.strip().startswith("[")]
                # Use a low-resolution mono analysis of the first track for a lightweight spectrum overlay.
                stft = None
                if album_show_spectrum:
                    yy, sr = librosa.load(temp_paths_album[0], sr=11025, mono=True)
                    stft = np.abs(librosa.stft(yy, n_fft=512, hop_length=max(128, int(11025 / fps))))

                def fit_frame(img):
                    img = img.convert("RGB")
                    sw, sh = img.size
                    scale = max(w / max(sw, 1), h / max(sh, 1))
                    img = img.resize((max(w, int(sw * scale)), max(h, int(sh * scale))), Image.Resampling.LANCZOS)
                    left, top = (img.width-w)//2, (img.height-h)//2
                    return img.crop((left, top, left+w, top+h))

                cycle_duration = sum(v[2] for v in visual_clips_album)
                def make_frame(t):
                    vt = float(t) % cycle_duration
                    idx = 0
                    for j, item in enumerate(visual_clips_album):
                        if vt < item[2]: idx = j; break
                        vt -= item[2]
                    typ, obj, dur = visual_clips_album[idx]
                    if typ == "video":
                        arr = obj.get_frame(min(vt, max(0, dur-0.04)))
                        img = Image.fromarray(np.asarray(arr).astype(np.uint8)).convert("RGB")
                    else: img = obj.copy()
                    img = fit_frame(img)
                    draw = ImageDraw.Draw(img, "RGBA")
                    if album_show_lyrics and lyrics:
                        line = lyrics[min(len(lyrics)-1, int(t / max(total_duration, .1) * len(lyrics)))]
                        try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(24, h//25))
                        except Exception: font = ImageFont.load_default()
                        max_width = int(w*.88)
                        while len(line) > 8 and draw.textbbox((0,0), line, font=font)[2] > max_width: line = line[:-2].rstrip()+"…"
                        box = draw.textbbox((0,0), line, font=font, stroke_width=2)
                        tw, th = box[2]-box[0], box[3]-box[1]
                        x, y = (w-tw)//2, int(h*.79)
                        px, py = max(12,w//90), max(8,h//90)
                        draw.rounded_rectangle((x-px,y-py,x+tw+px,y+th+py), radius=12, fill=(0,0,0,155))
                        draw.text((x,y), line, font=font, fill=(255,255,255,255), stroke_width=2, stroke_fill=(0,0,0,220))
                    if album_show_spectrum and stft is not None and stft.shape[1]:
                        col = min(stft.shape[1]-1, int(t*fps) % stft.shape[1])
                        vals = stft[:,col]
                        count, gap = 40, max(3,w//320)
                        step = max(1,len(vals)//count)
                        bins = [float(np.mean(vals[k:k+step])) for k in range(0,len(vals),step)][:count]
                        peak = max(bins) if bins else 1
                        bw = max(3,(w-2*(w//16)-gap*(count-1))//count); x0=w//16; base=int(h*.93)
                        for bi,val in enumerate(bins):
                            bh=max(3,int(val/max(peak,1e-6)*h*.085)); x=x0+bi*(bw+gap)
                            draw.rounded_rectangle((x,base-bh,x+bw,base), radius=max(1,bw//3), fill=(40,220,255,180))
                    return np.asarray(img)

                final_clip_album = VideoClip(make_frame, duration=total_duration).set_audio(combined_audio_album)
                output_path = "album_video_output.mp4"
                final_clip_album.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac", preset="ultrafast", threads=2, bitrate="3500k" if w == 1920 else "1800k", logger="bar")
                st.success(f"Selesai: {len(album_audio_files)} audio, {len(album_visual_files)} visual, durasi {total_duration/60:.1f} menit.")
                with open(output_path, "rb") as f:
                    st.download_button("⬇️ Download Video Album MP4", f.read(), file_name="album_video.mp4", mime="video/mp4", key="download_album_multi")
        except Exception as e:
            st.error(f"Render album gagal: {type(e).__name__}: {e}")
        finally:
            try:
                if final_clip_album is not None: final_clip_album.close()
            except Exception: pass
            try:
                if combined_audio_album is not None: combined_audio_album.close()
            except Exception: pass
            for c in audio_clips_album:
                try: c.close()
                except Exception: pass
            for item in visual_clips_album:
                if item[0] == "video":
                    try: item[1].close()
                    except Exception: pass
            for path in temp_paths_album:
                try:
                    if os.path.exists(path): os.remove(path)
                except Exception: pass


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
    uploaded_bg1 = st.file_uploader("2. Artwork Utama / 1 (.jpg/.png/.mp4):", type=["jpg", "jpeg", "png", "mp4"])

with col_v3:
    uploaded_bg2 = st.file_uploader("3. Artwork 2 (Opsional — .jpg/.png/.mp4):", type=["jpg", "jpeg", "png", "mp4"])

with col_v4:
    uploaded_bg3 = st.file_uploader("4. Artwork 3 (Opsional — .jpg/.png/.mp4):", type=["jpg", "jpeg", "png", "mp4"])

st.caption("Setiap slot menerima JPG/PNG atau MP4 bergerak. Video artwork mengikuti durasi audio; jika lebih pendek, videonya diulang. Audio dari artwork MP4 diabaikan—audio utama tetap dari MP3/WAV. Bisa mencampur video dan gambar hingga 3 artwork.")

uploaded_logo = st.file_uploader("Upload Logo PNG Transparan (Opsional):", type=["png"])
uploaded_srt = st.file_uploader("Upload file lirik/subtitle .SRT (opsional — digunakan untuk render Long dan Short):", type=["srt"], help="Timestamp SRT menentukan kapan baris lirik tampil. Jika tidak diunggah, Long memakai pembagian lirik otomatis; Short tetap memakai hook dan CTA.")

col_opt1, col_opt2, col_opt3 = st.columns(3)
with col_opt1:
    style_spectrum = st.selectbox(
        "Gaya Spektrum Audio:",
        ["🚫 Tanpa Spektrum (Hanya Background + Logo)", "Balok (Equalizer)", "Bar (Solid)", "Line (Garis Wave)", "CLine (Circle Line)", "P2P (Point to Point)"]
    )
with col_opt2:
    color_themes = st.multiselect(
        "Warna Spektrum / Aksen (bisa pilih beberapa):",
        ["Neon Cyan", "Sunset Red", "Electric Purple", "Cyber Green", "Gold Sunset"],
        default=["Neon Cyan", "Electric Purple"],
        help="Pilih satu atau beberapa palet. Warna akan dipadukan pada spektrum dan partikel."
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

                # Artwork dapat berupa gambar statis atau video MP4 bergerak.
                # Semua klip artwork hanya dipakai sebagai visual; audio utamanya dari MP3/WAV.
                bg_assets = []
                available_bg_assets = []
                artwork_video_clips = []
                artwork_temp_paths = []

                def load_artwork_asset(uploaded_file, asset_index):
                    if uploaded_file is None:
                        return None
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    if ext in (".jpg", ".jpeg", ".png"):
                        img = Image.open(uploaded_file).convert("RGB")
                        return ("image", img)
                    if ext == ".mp4":
                        temp_video_path = f"temp_artwork_video_{asset_index}.mp4"
                        with open(temp_video_path, "wb") as temp_video_file:
                            temp_video_file.write(uploaded_file.getbuffer())
                        artwork_temp_paths.append(temp_video_path)
                        art_clip = VideoFileClip(temp_video_path, audio=False)
                        artwork_video_clips.append(art_clip)
                        if not art_clip.duration or art_clip.duration <= 0:
                            raise ValueError(f"Artwork MP4 {uploaded_file.name} tidak memiliki durasi video yang valid.")
                        return ("video", art_clip)
                    raise ValueError(f"Format artwork tidak didukung: {uploaded_file.name}")

                for asset_index, uploaded_art in enumerate((uploaded_bg1, uploaded_bg2, uploaded_bg3), start=1):
                    asset = load_artwork_asset(uploaded_art, asset_index)
                    if asset is not None:
                        bg_assets.append(asset)
                        available_bg_assets.append(asset)
                if not bg_assets:
                    raise ValueError("Unggah minimal satu artwork JPG, PNG, atau MP4.")

                def artwork_frame(asset, time_s, progress=0.0, apply_zoom=True):
                    asset_type, asset_data = asset
                    if asset_type == "video":
                        clip_duration = max(float(asset_data.duration), 0.001)
                        source_t = float(time_s) % clip_duration
                        frame_array = asset_data.get_frame(source_t)
                        frame_img = Image.fromarray(np.asarray(frame_array).astype(np.uint8)).convert("RGB")
                        # Video artwork mempertahankan gerakan asli dan tidak diberi zoom tambahan.
                        sw, sh = frame_img.size
                        scale = max(target_w / max(sw, 1), target_h / max(sh, 1))
                        resized = frame_img.resize((max(target_w, int(sw * scale)), max(target_h, int(sh * scale))), Image.Resampling.LANCZOS)
                        left = max(0, (resized.width - target_w) // 2)
                        top = max(0, (resized.height - target_h) // 2)
                        return resized.crop((left, top, left + target_w, top + target_h))

                    image = asset_data
                    if image.size != (target_w, target_h):
                        sw, sh = image.size
                        scale = max(target_w / max(sw, 1), target_h / max(sh, 1))
                        resized = image.resize((max(target_w, int(sw * scale)), max(target_h, int(sh * scale))), Image.Resampling.LANCZOS)
                        left = max(0, (resized.width - target_w) // 2)
                        top = max(0, (resized.height - target_h) // 2)
                        image = resized.crop((left, top, left + target_w, top + target_h))
                    if apply_zoom:
                        scale = 1.0 + (0.10 * max(0.0, min(1.0, progress)))
                        crop_w = max(1, int(target_w / scale))
                        crop_h = max(1, int(target_h / scale))
                        left = max(0, (target_w - crop_w) // 2)
                        top = max(0, (target_h - crop_h) // 2)
                        image = image.crop((left, top, left + crop_w, top + crop_h)).resize((target_w, target_h), Image.Resampling.LANCZOS)
                    return image

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

                # Gabungkan warna dari semua palet terpilih, tanpa mengubah aksen logo.
                if not color_themes:
                    color_themes = ["Neon Cyan"]
                palette_colors = []
                for selected_theme in color_themes:
                    for palette_color in get_theme_colors(selected_theme):
                        if palette_color not in palette_colors:
                            palette_colors.append(palette_color)
                if not palette_colors:
                    palette_colors = [(0, 220, 255), (0, 150, 255), (0, 255, 200)]
                primary_color = palette_colors[0]
                secondary_color = palette_colors[min(1, len(palette_colors) - 1)]
                accent_color = palette_colors[min(2, len(palette_colors) - 1)]

                def make_frame(t):
                    frame_idx = min(int(t * fps), stft.shape[1] - 1)
                    progress = t / duration

                    # Bagi durasi hanya ke artwork yang benar-benar diunggah.
                    segment_len = duration / max(1, len(bg_assets))
                    segment_idx = min(int(t / max(segment_len, 0.001)), len(bg_assets) - 1)
                    local_t = max(0.0, t - segment_idx * segment_len)
                    active_bg = bg_assets[segment_idx]
                    frame_bg = artwork_frame(active_bg, local_t, progress=progress, apply_zoom=True)

                    if effect_transitions and len(bg_assets) > 1:
                        fade_len = min(1.0, segment_len * 0.12)
                        if segment_idx > 0 and local_t < fade_len:
                            prev_bg = bg_assets[segment_idx - 1]
                            prev_local_t = max(0.0, segment_len - fade_len + local_t)
                            prev_frame = artwork_frame(prev_bg, prev_local_t, progress=progress, apply_zoom=True)
                            alpha = min(1.0, max(0.0, local_t / max(fade_len, 0.001)))
                            frame_bg = Image.blend(prev_frame, frame_bg, alpha)

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
                                    color_idx = min(int(seg_progress * len(palette_colors)), len(palette_colors) - 1)
                                    color = palette_colors[color_idx]
                                    draw.rectangle([x0, y0_seg, x1, y1_seg], fill=color)

                        elif style_spectrum == "Bar (Solid)":
                            for b in range(num_bars):
                                h_val = int(norm_amps[b] * max_bar_h) + 4
                                x0 = start_x + b * (bar_w + bar_gap)
                                x1 = x0 + bar_w
                                y0 = base_y - h_val
                                draw.rectangle([x0, y0, x1, base_y], fill=palette_colors[b % len(palette_colors)])

                        elif style_spectrum == "Line (Garis Wave)":
                            points = []
                            for b in range(num_bars):
                                x = start_x + b * (bar_w + bar_gap) + (bar_w // 2)
                                y = base_y - int(norm_amps[b] * max_bar_h)
                                points.append((x, y))
                            if len(points) > 1:
                                for pi in range(len(points) - 1):
                                    draw.line([points[pi], points[pi + 1]], fill=palette_colors[pi % len(palette_colors)], width=5)

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
                                for pi in range(len(circle_points) - 1):
                                    draw.line([circle_points[pi], circle_points[pi + 1]], fill=palette_colors[pi % len(palette_colors)], width=5)

                        elif style_spectrum == "P2P (Point to Point)":
                            points = []
                            for b in range(num_bars):
                                x = start_x + b * (bar_w + bar_gap) + (bar_w // 2)
                                y = base_y - int(norm_amps[b] * max_bar_h)
                                points.append((x, y))
                                point_color = palette_colors[b % len(palette_colors)]
                                draw.ellipse([x-5, y-5, x+5, y+5], fill=point_color)
                            if len(points) > 1:
                                for pi in range(len(points) - 1):
                                    draw.line([points[pi], points[pi + 1]], fill=palette_colors[pi % len(palette_colors)], width=4)

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
                        particle_count = 28 if is_short else 22
                        for pi in range(particle_count):
                            seed = (pi * 7919) % 997
                            px = int((seed * 37 + t * (18 + (pi % 7) * 5)) % target_w)
                            py = int(((seed * 53) - t * (12 + (pi % 5) * 4)) % target_h)
                            # Gelembung air lebih besar, transparan, dengan garis tepi dan kilau kecil.
                            radius = 6 + (pi % 5) * 2
                            alpha = 35 + (pi % 4) * 12
                            bubble_color = palette_colors[pi % len(palette_colors)]
                            pd.ellipse([px-radius, py-radius, px+radius, py+radius],
                                       fill=bubble_color + (alpha,),
                                       outline=bubble_color + (min(alpha + 100, 210),), width=2)
                            # Pantulan kecil di sisi kiri atas agar terasa seperti gelembung.
                            highlight_r = max(2, radius // 4)
                            pd.ellipse([px-radius//2, py-radius//2, px-radius//2+highlight_r,
                                        py-radius//2+highlight_r], fill=(255, 255, 255, 180))
                        soft_particles = particle_layer.filter(ImageFilter.GaussianBlur(radius=3))
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
                                "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
                                "/usr/share/fonts/truetype/clear-sans/ClearSans-Bold.ttf",
                                "/usr/share/fonts/truetype/andika/Andika-Bold.ttf",
                            ] if bold else [
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                                "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
                            ]
                        )
                        for font_path in font_paths:
                            try:
                                return ImageFont.truetype(font_path, size=size)
                            except (OSError, IOError):
                                pass
                        # Jangan jatuh ke font mungil bawaan Pillow jika font sistem tidak ditemukan.
                        try:
                            return ImageFont.load_default(size=size)
                        except TypeError:
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
                            # HOOK SHORT: ukuran dan gaya mengikuti contoh pengguna.
                            # Font kuning besar, panel hitam tegas, posisi tengah-atas artwork.
                            hook_label = re.sub(r"\s+", " ", str(auto_hook_text)).strip().upper()
                            max_text_width = int(target_w * 0.91)
                            # Ukuran acuan sekitar 6.5% lebar video (70 px pada 1080p).
                            hook_size = max(54, int(target_w * 0.065))
                            hook_font = load_font(hook_size)
                            hook_lines = wrap_text(hook_label, hook_font, max_text_width)
                            # Jika kalimat panjang, kecilkan secara bertahap agar tetap terbaca;
                            # maksimal dua baris, bukan dibiarkan menjadi tiga baris kecil.
                            while len(hook_lines) > 2 and hook_size > 48:
                                hook_size -= 3
                                hook_font = load_font(hook_size)
                                hook_lines = wrap_text(hook_label, hook_font, max_text_width)
                            hook_lines = hook_lines[:2]
                            hook_text = "\n".join(hook_lines)
                            text_bbox = draw.multiline_textbbox(
                                (0, 0), hook_text, font=hook_font, spacing=8, stroke_width=2
                            )
                            text_w = text_bbox[2] - text_bbox[0]
                            text_h = text_bbox[3] - text_bbox[1]
                            pad_x, pad_y = 30, 18
                            box_w = min(target_w - 36, text_w + pad_x * 2)
                            box_h = text_h + pad_y * 2
                            box_x = (target_w - box_w) // 2
                            # Sekitar 45% tinggi video seperti contoh, tanpa menutup wajah utama.
                            box_y = int(target_h * 0.445) - box_h // 2
                            box_y = max(100, min(box_y, target_h - box_h - 180))
                            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                            overlay_draw = ImageDraw.Draw(overlay)
                            overlay_draw.rounded_rectangle(
                                [box_x, box_y, box_x + box_w, box_y + box_h],
                                radius=8, fill=(0, 0, 0, 205)
                            )
                            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
                            draw = ImageDraw.Draw(canvas)
                            text_x = (target_w - text_w) // 2
                            text_y = box_y + pad_y - text_bbox[1]
                            draw.multiline_text(
                                (text_x, text_y), hook_text, fill=(255, 220, 35), font=hook_font,
                                spacing=8, stroke_width=2, stroke_fill=(0, 0, 0)
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
                        (asset, auto_hooks[min(i, len(auto_hooks) - 1)], f"short_{i+1:02d}.mp4")
                        for i, asset in enumerate(available_bg_assets)
                    ]
                else:
                    render_jobs = [(None, auto_hook_text, "short_video.mp4" if is_short else "long_video.mp4")]

                rendered_files = []
                for job_index, (job_image, job_hook, output_name) in enumerate(render_jobs, start=1):
                    if job_image is not None:
                        bg_assets = [job_image]
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
                for artwork_clip in artwork_video_clips:
                    try:
                        artwork_clip.close()
                    except Exception:
                        pass
                for artwork_temp_path in artwork_temp_paths:
                    try:
                        if os.path.exists(artwork_temp_path):
                            os.remove(artwork_temp_path)
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

