# VisioMatch  
### AI Hybrid Search Engine & Visual Product Discovery
![VisioMatch Banner](https://img.shields.io/badge/AI-Vision-blueviolet?style=for-the-badge) ![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**VisioMatch** is an advanced AI-powered image similarity and visual search engine. It goes beyond simple embeddings by implementing a **Hybrid Search Architecture** that combines local deep learning models with global web search capabilities.

> **Key Innovation**: If the system detects that an uploaded image is "Out-of-Distribution" (not in the local catalog), it automatically fails over to a global search (via Google Lens/ImgBB) to provide relevant results regardless of the local dataset limitations.

---

## Key Features

### Core Intelligence
- **ResNet50 Backbone**: Uses a pre-trained ResNet50 model (truncated at the penultimate layer) to extract rich 2048-dimensional visual feature vectors.
- **Vector Similarity Search**: Powered by **FAISS (Facebook AI Similarity Search)** for ultra-fast L2 distance calculations and retrieval.
- **Out-of-Distribution (OOD) Detection**: Smart rejection thresholding (default: 190.0) identifies when an image doesn't belong to the known catalog.

### Hybrid Failover System
- **Local Catalog Search**: High-speed retrieval from the local Stanford Online Products dataset.
- **Global Discovery Mode**: Automatically triggers when local confidence is low. Uploads the image to **ImgBB** and redirects to **Google Lens** for a world-wide search.

### Modern Interactive UI
- **Streamlit-based Interface**: Fully interactive web application.
- **Dynamic Animations**: Custom CSS for glassmorphism, floating animations, and shimmer effects.
- **Lottie Integrations**: Visual feedback during deep scanning operations.

---

## Tech Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Core Logic** | Python 3.10+ | Primary programming language |
| **DL Framework** | PyTorch, Torchvision | Feature extraction (ResNet50) |
| **Indexing** | FAISS (Meta) | High-performance vector clustering & search |
| **Frontend** | Streamlit | Web UI & Interaction logic |
| **Cloud/API** | ImgBB API | Image hosting for global search capability |
| **Discovery** | Google Lens | External visual search engine |

---

## Project Structure

```bash
VisioMatch/
├── data/
│   ├── Stanford_Online_Products/  # Raw images
│   └── embeddings/                # Saved FAISS index & path pickles
├── src/
│   ├── model.py                   # ResNet50 definitions
│   └── dataset.py                 # PyTorch Dataset class
├── app.py                         # Main Streamlit Application
├── run_indexing.py                # Offline embedding generation script
├── requirements.txt               # Dependencies
└──  README.md                      # Documentation
```

---

## Setup & Installation

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/VisioMatch.git
cd VisioMatch
pip install -r requirements.txt
```

### 2. Configure Secrets (Crucial for Hybrid Search)
You must set up your API keys for the external search to work.
1. Create a file named names `.secrets.toml` inside the `.streamlit` folder: `.streamlit/secrets.toml`.
2. Add your **ImgBB API Key** (get one valid key from [api.imgbb.com](https://api.imgbb.com/)):

```toml
# .streamlit/secrets.toml
IMGBB_API_KEY = "your_api_key_here"
```

### 3. Run Indexing (First Time Only)
Generate the vector embeddings for your dataset.
```bash
python run_indexing.py
```
*Note: This will save `index.faiss` and `paths.pkl` in the `data/embeddings` directory.*

### 4. Launch the App
```bash
streamlit run app.py
```

---

##  Future Roadmap
- [ ] Integration with Qdrant/Milvus for persistent vector storage.
- [ ] Support for Text-to-Image search (CLIP integration).
- [ ] Docker containerization for easy deployment.

---
**Author**: [Jaya Sri Vardhan Samgoju]  
**License**: MIT
