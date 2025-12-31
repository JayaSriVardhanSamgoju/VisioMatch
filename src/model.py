import torch
import torch.nn as nn
from torchvision import models

class ImageEmbeddingModel(nn.Module):
    def __init__(self, model_name='resnet50'):
        super(ImageEmbeddingModel, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        
       
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            
            features = self.backbone(x)
            
            return torch.flatten(features, 1)