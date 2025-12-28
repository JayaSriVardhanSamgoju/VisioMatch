# VisioMatch  
### AI Image Similarity & Visual Search Engine

**VisioMatch** is an AI-powered image similarity system that retrieves visually similar images using deep CNN embeddings and vector similarity search.  
It demonstrates how modern visual search engines are built using deep learning and scalable retrieval techniques.

---

## Features
- CNN-based feature extraction  
- Offline embedding generation  
- Fast similarity search using FAISS  
- Top-K visual retrieval  
- Streamlit-based interactive UI  

---

## Pipeline
Image → CNN Embedding → Similarity Search → Top-K Results


---

## Dataset
- Stanford Online Products Dataset (SOP)

---

## Tech Stack
- Python, PyTorch, FAISS, Streamlit

---

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
