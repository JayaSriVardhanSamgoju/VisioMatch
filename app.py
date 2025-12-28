import streamlit as st
import torch
import faiss
import pickle
import os
from PIL import Image
from torchvision import transforms
import streamlit.components.v1 as components
from src.model import ImageEmbeddingModel

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="VisioMatch | AI Image Similarity",
    page_icon="🔍",
    layout="wide"
)

# =========================================================
# ADVANCED CSS (PROFESSIONAL ANIMATIONS)
# =========================================================
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
.stApp {
    background-color: #0B0E14;
    color: #E0E0E0;
}

/* ---------- HERO CONTAINER ---------- */
.hero-container {
    background: linear-gradient(135deg, #151A25, #0B0E14, #151A25);
    background-size: 300% 300%;
    animation: gradientFlow 12s ease infinite, heroFloat 1.2s ease-out forwards;
    padding: 60px 20px;
    border-radius: 30px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 40px;
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes heroFloat {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---------- SHIMMER LETTER REVEAL ---------- */
@keyframes shimmerReveal {
    from { opacity: 0; transform: translateY(20px); filter: blur(4px); }
    to { opacity: 1; transform: translateY(0); filter: blur(0); }
}

@keyframes shimmerGlow {
    50% { text-shadow: 0 0 14px rgba(255,75,75,0.9); }
}

@keyframes sparkle {
    50% {
        text-shadow: 0 0 10px white, 0 0 24px #FF4B4B;
        transform: scale(1.15);
    }
}

.app-title span {
    display: inline-block;
    opacity: 0;
    animation: shimmerReveal 0.6s ease forwards, shimmerGlow 1.2s ease;
}

.app-title span:last-child {
    animation: shimmerReveal 0.6s ease forwards, sparkle 1s ease 0.2s;
}

/* ---------- SUBTITLE ---------- */
.app-subtitle {
    font-size: 1.2rem;
    color: #8B949E;
    margin-top: 12px;
    opacity: 0;
    animation: shimmerReveal 1.2s ease forwards;
    animation-delay: 1.8s;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background-color: #FF4B4B !important;
    border-radius: 14px !important;
    font-weight: bold !important;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 18px rgba(255,75,75,0.6);
}

/* ---------- PRODUCT CARD ---------- */
.product-card {
    background: #161B22;
    border-radius: 20px;
    padding: 15px;
    border: 1px solid #30363D;
    opacity: 0;
    animation: cardFadeUp 0.6s ease forwards;
}

@keyframes cardFadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.product-card:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 30px rgba(255,75,75,0.25);
}

/* ---------- PROGRESS BAR ---------- */
div[data-testid="stProgress"] > div {
    animation: progressFill 1.4s ease forwards;
}

@keyframes progressFill {
    from { width: 0%; }
    to { width: 100%; }
}

html { scroll-behavior: smooth; }

</style>
""", unsafe_allow_html=True)

# =========================================================
# BACKEND LOADING
# =========================================================
@st.cache_resource
def load_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageEmbeddingModel().to(device).eval()
    index = faiss.read_index("data/embeddings/index.faiss")

    with open("data/embeddings/paths.pkl", "rb") as f:
        paths = pickle.load(f)

    return model, index, paths, device

model, index, paths, device = load_engine()

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero-container">
    <div class="app-title" style="font-size:3.8rem;font-weight:900;color:#FF4B4B;">
        <span style="animation-delay:0s">V</span>
        <span style="animation-delay:0.1s">i</span>
        <span style="animation-delay:0.2s">s</span>
        <span style="animation-delay:0.3s">i</span>
        <span style="animation-delay:0.4s">o</span>
        <span style="animation-delay:0.5s">M</span>
        <span style="animation-delay:0.6s">a</span>
        <span style="animation-delay:0.7s">t</span>
        <span style="animation-delay:0.8s">c</span>
        <span style="animation-delay:0.9s">h</span>
    </div>
    <div class="app-subtitle">
        AI-Powered Image Similarity & Visual Search Engine
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SEARCH UI
# =========================================================
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📷 Upload Query Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, width=260)

with col2:
    k = st.select_slider("Results", options=[4, 8, 12, 16], value=8)
    search_btn = st.button("START VISIOMATCH SEARCH")

st.markdown("<div id='results'></div>", unsafe_allow_html=True)

# =========================================================
# SEARCH EXECUTION
# =========================================================
if uploaded_file and search_btn:
    components.html("""
        <script>
            document.getElementById('results').scrollIntoView({behavior:'smooth'});
        </script>
    """, height=0)

    img = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    with torch.no_grad():
        vec = model(transform(img).unsqueeze(0).to(device)).cpu().numpy().astype("float32")

    distances, indices = index.search(vec, k)

    st.divider()
    st.subheader("🎯 Similar Visual Matches")

    DATA_ROOT = "data/Stanford_Online_Products"
    cols = st.columns(4)

    for i, idx in enumerate(indices[0]):
        with cols[i % 4]:
            path = os.path.join(DATA_ROOT, paths[idx])
            score = max(0, 100 - distances[0][i] * 10)

            st.markdown("<div class='product-card'>", unsafe_allow_html=True)
            st.image(Image.open(path), use_container_width=True)
            st.markdown(f"**Similarity Score: {score:.1f}%**")
            st.progress(int(score))
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image to start visual similarity search.")

