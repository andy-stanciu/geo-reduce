import torch.nn as nn
import timm

from city_mapping import NUM_CITIES


class ViTCityClassifier(nn.Module):
    """ViT for city classification (50 classes)"""
    
    def __init__(self, model_name='vit_base_patch16_224', 
                 pretrained=True, freeze_backbone=True, dropout=None):
        super().__init__()

        if dropout is None:
            raise ValueError('Dropout not provided')
        
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.vit.head.in_features
        
        # Freeze backbone (highly recommended for classification)
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Classification head
        self.vit.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CITIES)
        )
        
        # Ensure head is trainable
        for param in self.vit.head.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.vit(x)  # Returns logits [batch, 50]
