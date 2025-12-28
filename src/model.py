import torch
import torch.nn as nn
from torchvision import models

class ImageEmbeddingModel(nn.Module):
    def __init__(self, model_name='resnet50'):
        super(ImageEmbeddingModel, self).__init__()
        # Load weights: Stanford's CS231n suggests starting with ImageNet weights
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        
        # We strip the final FC layer (classification head)
        # We keep everything up to the Global Average Pooling layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.eval() # Permanent evaluation mode

    def forward(self, x):
        with torch.no_grad():
            # Output from backbone is [Batch, 2048, 1, 1]
            features = self.backbone(x)
            # Flatten to [Batch, 2048]
            return torch.flatten(features, 1)