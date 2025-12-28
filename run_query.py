import torch
import faiss
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.model import ImageEmbeddingModel

def run_visual_query(query_image_path, top_k=5):
    # 1. Paths to your saved "Memory"
    INDEX_PATH = "data/embeddings/index.faiss"
    PATHS_PATH = "data/embeddings/paths.pkl"
    DATA_ROOT = "data/Stanford_Online_Products"

    # 2. Load the System State
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageEmbeddingModel().to(device)
    index = faiss.read_index(INDEX_PATH)
    
    with open(PATHS_PATH, 'rb') as f:
        stored_paths = pickle.load(f)

    # 3. Preprocess Query Image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    query_img = Image.open(query_image_path).convert('RGB')
    input_tensor = transform(query_img).unsqueeze(0).to(device)

    # 4. Extract Query Vector
    query_vector = model(input_tensor).cpu().numpy().astype('float32')

    # 5. Search the FAISS Index
    distances, indices = index.search(query_vector, top_k)

    # 6. Visualize Results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, top_k + 1, 1)
    plt.title("Query")
    plt.imshow(query_img)
    plt.axis('off')

    for i, idx in enumerate(indices[0]):
        result_path = os.path.join(DATA_ROOT, stored_paths[idx])
        result_img = Image.open(result_path)
        
        plt.subplot(1, top_k + 1, i + 2)
        plt.title(f"Match {i+1}")
        plt.imshow(result_img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test it with an image from your test set or a random download
    # Ensure you have an image at this path!
    test_image = "data/Stanford_Online_Products/cabinet_final/400972619616_6.JPG" 
    if os.path.exists(test_image):
        run_visual_query(test_image)
    else:
        print(f"Please place an image named '{test_image}' in the root folder to test.")