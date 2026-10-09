import streamlit as st
import google.generativeai as genai

# Setup Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Suno AI Song & Visual Architect", layout="wide", page_icon="🎵")
st.title("🎵 Suno AI Song, Visual & SEO Architect")
st.caption("Platform Produksi Lagu, Lirik High-Retention, Prompt Suno, Artwork Sinematik, & SEO Optimizer")

# ----------------------------------------------------
# CONFIGURATION & API KEY WITH AUTOMATIC MODEL FALLBACK
# ----------------------------------------------------
@st.cache_resource
def get_working_model():
    """Mencari model Gemini yang valid dan aktif dari akun API key"""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        
        # Daftar prioritas model yang akan dicoba secara berurutan
        preferred_models = [
            'gemini-1.5-flash',
            'gemini-2.0-flash',
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro',
            'gemini-1.0-pro'
        ]
        
        # Cek daftar model yang benar-benar didukung oleh API Key kamu
        available_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Pilih model pertama yang cocok
        for target in preferred_models:
            if target in available_models:
                return genai.GenerativeModel(target)
                
        # Jika tidak ada yang cocok di daftar preferred, pilih model pertama dari yang tersedia
        if available_models:
            return genai.GenerativeModel(available_models[0])
            
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"⚠️ Gagal menghubungkan Gemini API Key: {e}")
        return None

model = get_working_model()

if not model:
    st.error("⚠️ Pastikan `GEMINI_API_KEY` sudah dikonfigurasi dengan benar di Streamlit Secrets!")
    st.stop()

# ----------------------------------------------------
# 1. TEMA & SUB-TEMA (DINAMIS)
# ----------------------------------------------------
st.subheader("1. Pilih Tema & Sub-Tema")

dict_tema = {
    "Keluarga": ["Sosok Ayah", "Pengorbanan Ibu", "Cinta Istri", "Pesan untuk Anak"],
    "Perjuangan Hidup": ["Bangkit dari Kehancuran", "Kerja Keras Malam Hari", "Merantau", "Menembus Batas"],
    "Cinta & Hubungan": ["Setia Sampai Tua", "Rindu Tak Tersampaikan", "Dermaga Terakhir", "Cinta Bedha Alam"],
    "Kehilangan & Harapan": ["Kehilangan Tanpa Pamit", "Lentera di Tengah Gelap", "Mengikhlaskan"]
}

col1, col2 = st.columns(2)

with col1:
    tema_utama = st.selectbox("Pilih Tema Utama:", list(dict_tema.keys()))

with col2:
    sub_tema_options = ["(Opsional - Tanpa Sub-Tema)"] + dict_tema[tema_utama] + ["Custom Sub-Tema..."]
    sub_tema_pilihan = st.selectbox("Pilih Sub-Tema:", sub_tema_options)

if sub_tema_pilihan == "Custom Sub-Tema...":
    sub_tema_final = st.text_input("Tuliskan Sub-Tema Khususmu:")
elif sub_tema_pilihan == "(Opsional - Tanpa Sub-Tema)":
    sub_tema_final = ""
else:
    sub_tema_final = sub_tema_pilihan

# ----------------------------------------------------
# 2. GENERATE & PILIH JUDUL
# ----------------------------------------------------
st.divider()
st.subheader("2. Pilih Judul Lagu")

if 'daftar_judul' not in st.session_state:
    st.session_state.daftar_judul = []

if st.button("🔍 Cari Ide Judul Unik"):
    with st.spinner("Mencari judul estetik yang bebas pasaran..."):
        prompt_judul = f"""
        Buatkan 5 pilihan judul lagu unik, puitis, belum pernah dipakai orang lain, dan sangat menarik perhatian.
        Tema Utama: {tema_utama}
        Sub-Tema: {sub_tema_final}
        Output HANYA berupa daftar 5 judul tanpa angka atau kalimat pengantar, dipisahkan koma.
        """
        try:
            res = model.generate_content(prompt_judul)
            st.session_state.daftar_judul = [j.strip() for j in res.text.split(",") if j.strip()]
        except Exception as e:
            st.error(f"Error saat membuat judul: {e}")

if st.session_state.daftar_judul:
    judul_terpilih = st.radio("Pilih Judul yang Kamu Sukai:", st.session_state.daftar_judul)
else:
    judul_terpilih = st.text_input("Atau Tulis Judul Sendiri:", "Jejak Langkah Tanpa Bersuara")

# ----------------------------------------------------
# 3. PILIHAN MULTI-GENRE
# ----------------------------------------------------
st.divider()
st.subheader("3. Kombinasi Genre Musik (Multi-Select)")

daftar_genre = [
    "Heavy Metal", "Modern Dangdut", "Mistis / Gamelan", "Indonesian Pop Ballad",
    "Acoustic Folk", "EDM / Synthwave", "Orchestral Cinematic", "Hip-Hop / Rap",
    "Rock Gothic", "Reggae Sunset"
]

selected_genres = st.multiselect(
    "Pilih 1 atau beberapa genre untuk dicampur (Eksperimen Genre):",
    options=daftar_genre,
    default=["Heavy Metal", "Modern Dangdut", "Mistis / Gamelan"]
)

# ----------------------------------------------------
# 4. GENERATE LIRIK HIGH-RETENTION & SUNO PROMPT
# ----------------------------------------------------
st.divider()
st.subheader("4. Lirik & Prompt Suno AI")

if st.button("🔥 Buat Lirik & Style Suno (High Retention Hook)", type="primary"):
    if not selected_genres:
        st.warning("Pilih minimal 1 genre terlebih dahulu!")
    else:
        string_genre = ", ".join(selected_genres)
        
        with st.spinner("Merancang struktur lirik pemikat penonton..."):
            prompt_lirik = f"""
            Bertindaklah sebagai Songwriter & Music Producer Profesional Spesialis Suno AI.
            Buatkan struktur lagu lengkap dan Style Prompt untuk Suno AI berdasarkan parameter berikut:
            - Judul Lagu: {judul_terpilih}
            - Tema: {tema_utama} ({sub_tema_final})
            - Kombinasi Genre: {string_genre}

            CRITICAL INSTRUCTIONS FOR HIGH RETENTION & VIRAL HOOK:
            1. **Style of Music Prompt (Suno)**: Buat dalam bahasa Inggris, singkat, padat, dan mencakup gabungan genre {string_genre} secara presisi.
            2. **Intro Hook (3 Detik Pertama)**: Wajib berisi elemen mengejutkan/pemikat (misal Spoken word mistis, Bisikan, atau Drop kencang) agar pendengar langsung tertarik.
            3. **Struktur Lirik**:
               - Gunakan metatag resmi Suno AI: [Intro], [Hook], [Verse 1], [Pre-Chorus], [Chorus], [Drop], [Bridge], [Outro].
               - **Chorus / Reff**: Wajib memiliki rima yang kuat, kata-kata yang mudah dihafal (earworm), dan emosi puncak yang membuat orang ingin memutar ulang lagu (replay value).
               - Berikan petunjuk eksekusi vokal di dalam kurung siku, misal: [Aggressive Scream], [Soft Whispering], [Fast Kendang Beat].
            """
            try:
                response = model.generate_content(prompt_lirik)
                st.session_state.last_lirik_result = response.text
                st.success("Lirik & Style Berhasil Dibuat!")
            except Exception as e:
                st.error(f"Error saat membuat lirik: {e}")

if 'last_lirik_result' in st.session_state:
    st.markdown(st.session_state.last_lirik_result)

# ----------------------------------------------------
# 5. GENERATOR ARTWORK & VISUAL PROMPTS
# ----------------------------------------------------
st.divider()
st.subheader("🎨 5. Artwork & Visual Prompt Generator")

col_art1, col_art2 = st.columns(2)

with col_art1:
    art_style = st.selectbox(
        "Pilih Mode / Gaya Visual Artwork:",
        [
            "Dark Cinematic & Gritty (Atmospheric Shadow & Mood)",
            "Surrealist Mystical Fantasy (Glow & Ethereal)",
            "Oil Painting Expressionism (Artistic Texture)",
            "8K Ultra-Realistic Photography (Cinematic Lighting)",
            "Vintage Pencil & Ink Sketch (Classic Hand-Drawn)",
            "Cyberpunk Neon Futurology (High Contrast Tech)",
            "Minimalist Gothic Silhouette (Bold & Spooky)"
        ]
    )

with col_art2:
    aspect_ratios = st.multiselect(
        "Pilih Format Output Visual:",
        [
            "16:9 (Cover & Thumbnail Youtube)",
            "9:16 (Shorts/TikTok/Reels - 3 Visual Berbeda)"
        ],
        default=["16:9 (Cover & Thumbnail Youtube)", "9:16 (Shorts/TikTok/Reels - 3 Visual Berbeda)"]
    )

if st.button("🖼️ Generate Prompt Artwork Visual"):
    if not aspect_ratios:
        st.warning("Pilih minimal satu format aspect ratio!")
    else:
        with st.spinner("Merancang 3 visual Shorts & 1 Cover sinematik..."):
            prompt_art = f"""
            Bertindaklah sebagai Art Director Sinematik Profesional.
            Buatkan deskripsi Prompt Gambar dalam Bahasa Inggris (siap-copas untuk Midjourney v6, DALL-E 3, atau Imagen) berdasarkan:
            - Judul Lagu: {judul_terpilih}
            - Tema: {tema_utama} ({sub_tema_final})
            - Gaya Visual: {art_style}
            - Format Pilihan: {", ".join(aspect_ratios)}

            PERATURAN PENTING GENERASI VISUAL:
            1. DILARANG ADA TEKS/TULISAN APAPUN DI DALAM GAMBAR (Clean visual, no text, no typography).
            2. Jika memilih **16:9 (Cover/Thumbnail)**:
               - Buat 1 Prompt Landscape yang dramatis, komposisi kuat, focal point jelas, kontras tinggi yang sangat cocok untuk thumbnail YouTube.
            3. Jika memilih **9:16 (Shorts/TikTok - 3 Visual Berbeda)**:
               - Wajib buat **3 Konsep Prompt Vertikal Unik & Terpisah**:
                 * **Visual 1 (Subjek/Karakter Utama Focus)**: Medium/close-up shot karakter atau benda utama.
                 * **Visual 2 (Atmospheric Environment Focus)**: Wide shot suasana latar tempat yang megah & estetik.
                 * **Visual 3 (Dramatis / Metaphor Shot)**: Shot aksi, simbolis, atau pencahayaan ekstrem yang emosional.

            Berikan penjelasan singkat Bahasa Indonesia untuk tiap konsep, diikuti Prompt Bahasa Inggris yang tebal (bold).
            """
            try:
                res_art = model.generate_content(prompt_art)
                st.session_state.last_art_result = res_art.text
                st.success("Prompt Artwork Visual Siap Digunakan!")
            except Exception as e:
                st.error(f"Error saat membuat prompt visual: {e}")

if 'last_art_result' in st.session_state:
    st.markdown(st.session_state.last_art_result)

# ----------------------------------------------------
# 6. SEO OPTIMIZER (YOUTUBE & TIKTOK)
# ----------------------------------------------------
st.divider()
st.subheader("📈 6. YouTube & TikTok SEO Optimizer")

if st.button("🚀 Optimasi SEO untuk Release Lagu", type="primary"):
    string_genre = ", ".join(selected_genres) if selected_genres else "General Genre"
    lirik_context = st.session_state.get('last_lirik_result', 'Lirik belum dibuat.')
    
    with st.spinner("Menyusun strategi SEO YouTube & TikTok..."):
        prompt_seo = f"""
        Bertindaklah sebagai Digital Music Marketer & SEO Specialist Spesialis YouTube & TikTok Music.
        Buatkan paket SEO lengkap untuk lagu berikut:
        - Judul Lagu: {judul_terpilih}
        - Tema: {tema_utama} ({sub_tema_final})
        - Genre: {string_genre}
        - Konteks Lirik: {lirik_context[:500]}...

        FORMAT KELUARAN YANG DIBUTUHKAN:

        ### 1. Judul Video YouTube (3 Opsi Clickbait & High CTR)
        - Opsi 1: [Format Emosional / Menyentuh Hati]
        - Opsi 2: [Format Genre & Style Unik / Eksperimental]
        - Opsi 3: [Format Trending / Penasaran]

        ### 2. Deskripsi YouTube (Siap Copas)
        - Paragraf Pembuka yang mengandung keyword pencarian.
        - Timestamps / Chapter (00:00 Intro, 00:03 Hook, dst).
        - Bagian Lirik Singkat / Reff utama.
        - Call to Action (Subscribe, Like, Comment).

        ### 3. Paket Shorts & TikTok
        - **Caption/Deskripsi Pendek**: Menarik perhatian dalam 1 kalimat.
        - **15 Hashtags Target**: Gabungan hashtag populer & niche khusus (contoh: #SunoAI #MetalDangdut #LaguNostalgia).

        ### 4. YouTube Video Tags (Koma-separated)
        - Sediakan 15-20 kata kunci pencarian yang dipisahkan koma agar mudah langsung di-copy-paste ke kolom Tags di YouTube Studio.
        """
        try:
            res_seo = model.generate_content(prompt_seo)
            st.session_state.last_seo_result = res_seo.text
            st.success("Paket SEO Berhasil Dibuat!")
        except Exception as e:
            st.error(f"Error saat membuat SEO: {e}")

if 'last_seo_result' in st.session_state:
    st.markdown(st.session_state.last_seo_result)
