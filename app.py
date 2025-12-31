import streamlit as st
import torch
import faiss
import pickle
import os
import time
import requests
from PIL import Image
from torchvision import transforms
import streamlit.components.v1 as components
from src.model import ImageEmbeddingModel
from streamlit_lottie import st_lottie

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="VisioMatch AI | Professional Visual Search",
    page_icon="🔍",
    layout="wide"
)

# --- CALIBRATION SETTINGS ---
# Based on your test, 210.93 is an outlier. 190.0 is the safe limit.
REJECTION_THRESHOLD = 190.0 
DATA_ROOT = "data/Stanford_Online_Products"

# =========================================================
# ADVANCED CSS (ANIMATIONS & GLASSMORPHISM)
# =========================================================
st.markdown("""
<style>
.stApp { background-color: #0B0E14; color: #E0E0E0; }

.hero-container {
    background: linear-gradient(135deg, #151A25, #0B0E14, #151A25);
    background-size: 300% 300%;
    animation: gradientFlow 12s ease infinite, heroFloat 1.2s ease-out forwards;
    padding: 60px 20px; border-radius: 30px; text-align: center;
    border: 1px solid rgba(255,255,255,0.06); margin-bottom: 40px;
}

@keyframes gradientFlow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes heroFloat { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }

.app-title span { display: inline-block; opacity: 0; animation: shimmerReveal 0.6s ease forwards; }
@keyframes shimmerReveal { from { opacity: 0; transform: translateY(20px); filter: blur(4px); } to { opacity: 1; transform: translateY(0); filter: blur(0); } }

.product-card {
    background: #161B22; border-radius: 20px; padding: 15px; border: 1px solid #30363D;
    transition: all 0.3s ease; opacity: 0; animation: cardFadeUp 0.6s ease forwards;
}
@keyframes cardFadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.product-card:hover { transform: scale(1.05); box-shadow: 0 10px 30px rgba(255,75,75,0.25); border-color: #FF4B4B; }

html { scroll-behavior: smooth; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# ASSET & ENGINE LOADING
# =========================================================
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_scan = load_lottieurl("https://lottie.host/8018261a-0797-4886-9051-512140e4f447/B6V2iR7j1n.json")

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
# SIDEBAR (Calibration Visualizer)
# =========================================================
st.sidebar.header("🛠️ Engine Calibration")
show_metrics = st.sidebar.toggle("Show Distance Metrics", value=False)
user_threshold = st.sidebar.slider("Rejection Sensitivity", 100.0, 400.0, REJECTION_THRESHOLD)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown(f"""
<div class="hero-container">
    <div class="app-title" style="font-size:3.8rem;font-weight:900;color:#FF4B4B;">
        {''.join([f'<span style="animation-delay:{i*0.1}s">{char}</span>' for i, char in enumerate("VisioMatch")])}
    </div>
    <div class="app-subtitle" style="color:#8B949E; margin-top:10px;">Enterprise-Grade Neural Search Engine</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SEARCH INTERFACE
# =========================================================
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📷 Upload Reference Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, width=280, caption="User Query")

with col2:
    k = st.select_slider("Results Precision", options=[4, 8, 12, 16], value=8)
    search_btn = st.button("RUN DEEP VISUAL SCAN")

st.markdown("<div id='results'></div>", unsafe_allow_html=True)

# =========================================================
# SEARCH EXECUTION
# =========================================================
if uploaded_file and search_btn:
    components.html("""<script>window.parent.document.getElementById('results').scrollIntoView({behavior:'smooth'});</script>""", height=0)

    # UI Scanning Animation
    anim_placeholder = st.empty()
    with anim_placeholder.container():
        if lottie_scan:
            st_lottie(lottie_scan, height=280, key="scanning")
        else:
            st.spinner("🔍 Scanning neural features...")
        st.markdown("<p style='text-align:center;'>Calculating Euclidean Proximity...</p>", unsafe_allow_html=True)

    # 1. Processing
    img = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        vec = model(transform(img).unsqueeze(0).to(device)).cpu().numpy().astype("float32")

    # 2. Search
    distances, indices = index.search(vec, k)
    anim_placeholder.empty()

    # 3. VERIFICATION: Distance Rejection
    best_dist = distances[0][0]

    if best_dist > user_threshold:
        st.markdown("<br>", unsafe_allow_html=True)
        st.error("### 🚫 Input Not Recognized")
        st.markdown(f"""
            <div style="background-color: rgba(255, 75, 75, 0.1); padding: 25px; border-radius: 15px; border: 1px solid #FF4B4B;">
                <h4 style="color: #FF4B4B; margin: 0;">VisioMatch Security Protocol</h4>
                <p style="margin: 10px 0 0 0; color: #E0E0E0;">
                    The visual features of this image do not match our enterprise catalog. 
                    The system detected a distance of <b>{best_dist:.2f}</b>, which exceeds the safety threshold.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Success: Reveal Match Details
        img_rel_path = paths[indices[0][0]]
        category = img_rel_path.split('/')[0].replace('_', ' ').title()
        
        st.markdown(f"### Related images for your search: <span style='color:#FF4B4B;'>{category}</span>", unsafe_allow_html=True)
        
        if show_metrics:
            st.metric("Closest Match Score", f"{best_dist:.2f}", delta="Within Catalog Range")

        # 4. Results Grid
        cols = st.columns(4)
        for i, idx in enumerate(indices[0]):
            with cols[i % 4]:
                # Path Normalization for Windows
                full_path = os.path.normpath(os.path.join(DATA_ROOT, paths[idx]))
                
                # Confidence Score Calculation (Normalized for viewers)
                # 0 distance = 100%, 250 distance = 0%
                conf_score = max(0, 100 - (distances[0][i] / 2.5))

                st.markdown("<div class='product-card'>", unsafe_allow_html=True)
                if os.path.exists(full_path):
                    st.image(Image.open(full_path), use_container_width=True)
                    st.markdown("**Visual Match Found**")
                    st.progress(int(conf_score))
                else:
                    st.error("File Path Error")
                    st.caption(f"Missing: {paths[idx]}")
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("💡 VisioMatch is standing by. Upload a product image to begin neural verification.")