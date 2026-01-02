import streamlit as st
import torch
import faiss
import pickle
import os
import requests
from PIL import Image
from torchvision import transforms
import streamlit.components.v1 as components
from src.model import ImageEmbeddingModel
from streamlit_lottie import st_lottie
import base64 # Import for Base64 encoding
import io     # Import for BytesIO

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="VisioMatch AI | Hybrid Search Engine",
    page_icon="🔍",
    layout="wide"
)

# --- CONFIGURATION SETTINGS ---
# 190.0 is the strict limit based on your previous tests.
REJECTION_THRESHOLD = 190.0 
DATA_ROOT = "data/Stanford_Online_Products"
# Max image size for ImgBB free tier (20MB = 20 * 1024 * 1024 bytes)
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024 

# Load ImgBB API Key from Streamlit secrets
IMGBB_API_KEY = st.secrets.get("IMGBB_API_KEY") 
if not IMGBB_API_KEY:
    st.error("Error: ImgBB API Key not found in .streamlit/secrets.toml. Please add it.")
    st.stop() # Stop the app if API key is missing

# =========================================================
# ADVANCED CSS (ANIMATIONS & STYLING)
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
# BACKEND LOADING
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
# IMGBB UPLOAD FUNCTION
# =========================================================
def upload_image_to_imgbb(image_bytes: bytes, api_key: str) -> str | None:
    """Uploads an image to ImgBB and returns its public URL."""
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        st.error(f"Image too large! Max allowed is {MAX_IMAGE_SIZE_BYTES / (1024*1024):.0f}MB.")
        return None

    try:
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": api_key,
            "image": base64_image,
            "expiration": 600 # Image will expire in 10 minutes (600 seconds)
        }
        
        response = requests.post(url, data=payload, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        if result and result.get("success"):
            return result["data"]["url"]
        else:
            st.error(f"ImgBB upload failed: {result.get('error', {}).get('message', 'Unknown error')}")
            return None
    except requests.exceptions.Timeout:
        st.error("ImgBB upload timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error during ImgBB upload: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during ImgBB upload: {e}")
        return None

# =========================================================
# HERO SECTION
# =========================================================
st.markdown(f"""
<div class="hero-container">
    <div class="app-title" style="font-size:3.8rem;font-weight:900;color:#FF4B4B;">
        {''.join([f'<span style="animation-delay:{i*0.1}s">{char}</span>' for i, char in enumerate("VisioMatch")])}
    </div>
    <div class="app-subtitle" style="color:#8B949E; margin-top:10px;">End-to-End Neural Product Retrieval Engine</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SEARCH INTERFACE
# =========================================================
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "png", "jpeg"])
    # Store uploaded file bytes for ImgBB if needed later
    uploaded_file_bytes = None
    if uploaded_file:
        st.image(uploaded_file, width=280, caption="Reference Query")
        uploaded_file_bytes = uploaded_file.getvalue()

with col2:
    k = st.select_slider("Detection Depth", options=[4, 8, 12, 16], value=8)
    search_btn = st.button("EXECUTE VISIOMATCH SCAN")

st.markdown("<div id='results_view'></div>", unsafe_allow_html=True)

# =========================================================
# EXECUTION LOGIC (LOCAL -> FAILOVER)
# =========================================================
if uploaded_file and search_btn:
    # Smooth scroll to results
    components.html("""<script>window.parent.document.getElementById('results_view').scrollIntoView({behavior:'smooth'});</script>""", height=0)

    # 1. Scanning UI
    anim_placeholder = st.empty()
    with anim_placeholder.container():
        if lottie_scan:
            st_lottie(lottie_scan, height=280, key="scanning")
        else:
            st.spinner("🔍 Deep scanning visual features...")

    # 2. Online Inference
    img = Image.open(io.BytesIO(uploaded_file_bytes)).convert("RGB") # Use BytesIO for PIL
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        vec = model(transform(img).unsqueeze(0).to(device)).cpu().numpy().astype("float32")

    # 3. Vector Similarity Search
    distances, indices = index.search(vec, k)
    anim_placeholder.empty()

    best_dist = distances[0][0]

    # --- HYBRID LOGIC CHECK ---
    if best_dist > REJECTION_THRESHOLD:
        # THE "FAILOVER" STATE (Local Search failed)
        st.error("### 🚫 Product Not Recognized in Local Catalog")
        st.markdown(f"""
            <div style="background-color: rgba(255, 75, 75, 0.1); padding: 25px; border-radius: 15px; border: 1px solid #FF4B4B;">
                <h4 style="color: #FF4B4B; margin: 0;">VisioMatch Intelligence</h4>
                <p style="margin: 10px 0 10px 0; color: #E0E0E0;">
                    The detected visual distance (<b>{best_dist:.2f}</b>) is too high for our catalog. 
                    This item appears to be <b>Out-of-Distribution</b>.
                </p>
                <hr style="border: 0.5px solid rgba(255,255,255,0.1);">
                <p style="font-size: 0.9rem; color: #8B949E;">
                    <b>Global Discovery:</b> Attempting to search this image through the entire web via Google Lens...
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Attempt ImgBB upload for automatic Google Lens search
        with st.spinner("🚀 Uploading image for global search..."):
            public_img_url = upload_image_to_imgbb(uploaded_file_bytes, IMGBB_API_KEY)

        if public_img_url:
            lens_url = f"https://lens.google.com/uploadbyurl?url={public_img_url}"
            st.success("✅ Image uploaded. Redirecting to Google Lens for global matches!")
            st.markdown(f'<a href="{lens_url}" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: #4CAF50; color: white; border-radius: 8px; text-decoration: none; font-weight: bold;">View Global Matches Instantly</a>', unsafe_allow_html=True)
        else:
            st.warning("Could not automatically search Google Lens. Please try uploading to ImgBB manually or check your API key/image size.")


    else:
        # THE "FOUND" STATE (Local Search successful)
        img_rel_path = paths[indices[0][0]]
        category = img_rel_path.split('/')[0].replace('_', ' ').title()
        
        st.success(f"🎯 Visual Match Identified: {category} (Score: {best_dist:.2f})")

        cols = st.columns(4)
        for i, idx in enumerate(indices[0]):
            with cols[i % 4]:
                full_path = os.path.normpath(os.path.join(DATA_ROOT, paths[idx]))
                conf_score = max(0, 100 - (distances[0][i] / 2.5)) # Adjust scaling as needed

                st.markdown("<div class='product-card'>", unsafe_allow_html=True)
                if os.path.exists(full_path):
                    st.image(Image.open(full_path), use_container_width=True)
                    st.markdown("**Local Catalog Match**")
                    st.progress(int(conf_score))
                else:
                    st.error("File Path Error")
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("💡 VisioMatch is standing by. Upload a product image to begin neural verification.")