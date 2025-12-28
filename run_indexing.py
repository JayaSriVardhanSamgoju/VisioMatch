import torch
import faiss
import pickle
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import SOPDataset
from src.model import ImageEmbeddingModel

# Move configurations outside
DATA_ROOT = "data/Stanford_Online_Products"
SAVE_DIR = "data/embeddings"
SAVE_PATH = os.path.join(SAVE_DIR, "index.faiss")
META_PATH = os.path.join(SAVE_DIR, "paths.pkl")
BATCH_SIZE = 64

def main():
    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageEmbeddingModel().to(device)

    # 2. Setup Data Loading
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SOPDataset(DATA_ROOT, 'Ebay_train.txt', transform=transform)
    # Keeping num_workers=4 is fine NOW because of the __main__ block
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Extraction Loop
    embeddings = []
    image_paths = []

    print(f"Running extraction on {device}...")
    for imgs, paths in tqdm(loader):
        imgs = imgs.to(device)
        out = model(imgs).cpu().numpy()
        embeddings.append(out)
        image_paths.extend(paths)

    # 4. Create FAISS Index
    print("Building FAISS index...")
    all_vecs = np.vstack(embeddings).astype('float32')
    dimension = all_vecs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_vecs)

    # 5. Save everything
    faiss.write_index(index, SAVE_PATH)
    with open(META_PATH, 'wb') as f:
        pickle.dump(image_paths, f)

    print(f"Success! Saved {len(image_paths)} image vectors to {SAVE_PATH}")

# THIS IS THE CRITICAL PART FOR WINDOWS
if __name__ == '__main__':
    main()