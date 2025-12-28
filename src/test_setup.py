from torchvision import transforms
from src.dataset import SOPDataset # Assuming your file is in src/
import matplotlib.pyplot as plt

# 1. Define the paths based on your Kaggle download
ROOT_DIR = "data/Stanford_Online_Products"
METADATA_FILE = "Ebay_train.txt"

# 2. Define standard Stanford CS231n transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 3. Initialize the dataset (This is where the "Input" happens)
sop_data = SOPDataset(root_dir=ROOT_DIR, txt_file=METADATA_FILE, transform=data_transform)

# 4. Access the first item to verify
image, path = sop_data[0]

print(f"Successfully loaded image from: {path}")
print(f"Image tensor shape: {image.shape}")

# Optional: Visualize it
# plt.imshow(image.permute(1, 2, 0))
# plt.show()