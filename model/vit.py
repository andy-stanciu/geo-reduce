import torch
import torch.nn as nn
import timm
    

class ViTGeoRegressor(nn.Module):
    """ViT adapted for lat/long coordinate regression"""
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, 
                 freeze_backbone=False):
        super().__init__()
        
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.vit.head.in_features
        
        # Freeze backbone if specified (only train head)
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Replace head with regression head
        self.vit.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )
        
        # Always ensure head is trainable
        for param in self.vit.head.parameters():
            param.requires_grad = True
        
    def forward(self, x, clip_output=False):
        output = self.vit(x)
        
        # Clip to [0, 1] at inference time only
        if clip_output and not self.training:
            output = torch.clamp(output, 0.0, 1.0)
        
        return output