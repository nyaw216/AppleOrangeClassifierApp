import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple vs Orange Classifier",
    page_icon="🍎",
    layout="wide"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Result box */
    .result-apple {
        background: linear-gradient(135deg, #ff6b6b18, #ff6b6b35);
        border: 2px solid #ff6b6b;
        border-radius: 16px;
        padding: 1.8rem 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #E74C3C;
        margin: 0.8rem 0;
    }
    .result-orange {
        background: linear-gradient(135deg, #ffa50018, #ffa50035);
        border: 2px solid #ffa500;
        border-radius: 16px;
        padding: 1.8rem 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #E67E22;
        margin: 0.8rem 0;
    }
    .info-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #e2e8f0;
        font-size: 0.9rem;
        line-height: 1.7;
        margin-bottom: 0.8rem;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ff6b6b, #ffa500);
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
# Catatan: @st.cache_resource memastikan model hanya di-load 1x, bukan setiap interaksi user
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model_apple_orange.h5")
    return model

model = load_model()

# ── KONSTANTA ──────────────────────────────────────────────────────────────────
# Sesuai notebook Week 2:
# - IMG_SIZE = (224, 224) → resolusi standar MobileNetV2
# - class_indices = {'apple': 0, 'orange': 1} → output sigmoid < 0.5 = apple, ≥ 0.5 = orange
IMG_SIZE   = (224, 224)
CLASS_NAMES = {0: ("Apple", "🍎", "result-apple", "#E74C3C"),
               1: ("Orange", "🍊", "result-orange", "#E67E22")}

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.title("🍎 vs 🍊 Apple & Orange Classifier")
st.markdown("Klasifikasi gambar buah menggunakan **Custom CNN** yang dilatih dari nol pada dataset Apple vs Orange.")
st.divider()

# ── LAYOUT: 2 KOLOM ────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    st.subheader("📤 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih file gambar buah",
        type=["jpg", "jpeg", "png"],
        help="Upload foto apel atau jeruk. Pastikan buah terlihat jelas dan pencahayaan cukup."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Gambar: {uploaded_file.name}", use_column_width=True)

with right_col:
    st.subheader("🔮 Hasil Prediksi")

    if uploaded_file is None:
        st.info("👆 Upload gambar buah di sebelah kiri untuk memulai klasifikasi.")

        # Tampilkan info model di sidebar kanan saat belum ada gambar
        st.markdown("""
        <div class='info-card'>
        <b>📋 Tentang Model</b><br><br>
        • <b>Arsitektur:</b> Custom CNN (3 Conv Blocks)<br>
        • <b>Input:</b> 224 × 224 × 3 (RGB)<br>
        • <b>Output:</b> Sigmoid — binary classification<br>
        • <b>Kelas:</b> Apple (0) vs Orange (1)<br>
        • <b>Dataset:</b> Apple vs Orange (LAS Week 2)
        </div>
        """, unsafe_allow_html=True)
    else:
        # Preprocessing — identik dengan saat training
        # 1. Resize ke 224×224 (sesuai IMG_SIZE notebook Week 2)
        # 2. Normalisasi pixel ke [0, 1] via /255.0 (sesuai rescale=1./255 di ImageDataGenerator)
        # 3. Tambah batch dimension: (H, W, C) → (1, H, W, C)
        img_resized = image.resize(IMG_SIZE)
        img_array  = np.array(img_resized) / 255.0
        img_array  = np.expand_dims(img_array, axis=0)   # shape: (1, 224, 224, 3)

        with st.spinner("Menganalisis gambar... 🔍"):
            raw_pred   = model.predict(img_array, verbose=0)
            confidence = float(raw_pred[0][0])   # output sigmoid → nilai 0-1

        # Interpretasi: output < 0.5 = Apple (kelas 0), ≥ 0.5 = Orange (kelas 1)
        class_idx  = 1 if confidence >= 0.5 else 0
        conf_display = confidence if class_idx == 1 else (1 - confidence)
        name, emoji, css_cls, color = CLASS_NAMES[class_idx]

        # Tampilkan hasil
        st.markdown(f"""
        <div class='{css_cls}'>
            {emoji} {name}
        </div>
        """, unsafe_allow_html=True)

        st.metric("Confidence", f"{conf_display:.1%}")
        st.progress(conf_display)

        st.markdown("---")
        st.markdown("**Detail probabilitas:**")
        apple_prob  = 1 - confidence
        orange_prob = confidence

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🍎 Apple",  f"{apple_prob:.1%}",
                      delta=None,
                      help="Probabilitas gambar adalah apel")
        with col_b:
            st.metric("🍊 Orange", f"{orange_prob:.1%}",
                      delta=None,
                      help="Probabilitas gambar adalah jeruk")

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Informasi Aplikasi")
    st.markdown("""
    **LAS Big Data 2026 — Week 5**
    *Deployment Model*

    **Model yang dipakai:**
    - Arsitektur: Custom CNN
    - Week 2 — Apple vs Orange
    - Input size: 224 × 224

    **Cara kerja:**
    1. Upload foto buah
    2. Model memproses gambar
    3. Output: Apple atau Orange + confidence %

    **Preprocessing:**
    - Resize ke 224×224
    - Normalisasi pixel /255.0
    """)
    st.divider()
    st.markdown("""
    *Dibuat oleh:*
    **Naufal Hanif (2616)**
    LAS Big Data MBC 2026
    """)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🍎🍊 Apple vs Orange Classifier — LAS Big Data 2026 | Naufal Hanif | Week 5 Deployment")
