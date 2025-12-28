## VisioMatch
AI-Powered Image Similarity & Visual Search Engine

VisioMatch is an end-to-end image similarity system that retrieves visually and semantically similar images using deep CNN embeddings and vector similarity search.
The project demonstrates how modern visual search engines are built using deep learning and scalable retrieval techniques.

Key Features

Deep feature extraction using pretrained CNNs

Offline embedding generation for efficient retrieval

Fast similarity search using FAISS

Real-time Top-K image retrieval

Clean, modular, production-style codebase

Interactive Streamlit web interface

System Overview
Image → CNN Embedding → FAISS Index → Similarity Search → Top-K Results

Dataset

Stanford Online Products Dataset (SOP)

~120K product images across ~12K categories

Designed specifically for image similarity and retrieval tasks

Tech Stack

Python 3.10

PyTorch & Torchvision

FAISS

NumPy, Scikit-learn

Streamlit

Project Structure
image_similarity_engine/
├── app.py
├── src/model.py
├── data/embeddings/
├── data/Stanford_Online_Products/
├── requirements.txt
└── README.md

How It Works

Images are preprocessed and passed through a pretrained CNN

Feature embeddings are generated and stored offline

FAISS indexes embeddings for fast similarity search

A query image is embedded at runtime

Top-K similar images are retrieved based on vector distance

Use Cases

Visual product search (e-commerce)

Image recommendation systems

Duplicate image detection

Computer vision–based search engines

Run the App
pip install -r requirements.txt
streamlit run app.py
