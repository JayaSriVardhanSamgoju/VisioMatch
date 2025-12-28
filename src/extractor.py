import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load ResNet-50 with default ImageNet weights
        resnet = models.resnet50(weights='DEFAULT')
        # Extract all layers except the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.eval() 

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            # Flatten [batch, 2048, 1, 1] to [batch, 2048]
            return torch.flatten(features, 1)