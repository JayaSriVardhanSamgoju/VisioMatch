import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SOPDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        # Standard SOP format: image_id, class_id, super_class_id, path
        self.data = pd.read_csv(os.path.join(root_dir, txt_file), sep=' ')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_relative_path = self.data.iloc[idx, 3]
        img_path = os.path.join(self.root_dir, img_relative_path)
        
        # Load image safely
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_relative_path