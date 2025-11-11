import torch.nn as nn
import timm


class ViTGeoRegressor(nn.Module):
    """ViT adapted for lat/long coordinate regression"""
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, 
                 interpolate_pos_embed=False):
        super().__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        
        # Replace classification head with regression head
        # Get the input features to the original head
        in_features = self.vit.head.in_features
        
        # Remove classification head and replace with regression head
        self.vit.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Linear(512, 2)  # Output: [latitude, longitude]
        )
        
        self.interpolate_pos_embed = interpolate_pos_embed
        
    def forward(self, x):
        return self.vit(x)
    