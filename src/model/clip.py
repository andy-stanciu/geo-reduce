import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig

from data.city_mapping import NUM_CITIES
from constants import *


class StreetCLIPCityClassifier(nn.Module):
    """
    City classifier using StreetCLIP pretrained vision encoder.
    
    StreetCLIP is pretrained on 1.1M street-view images for geolocation,
    making it much better than ImageNet for this task.
    """
    
    def __init__(self, model_name="geolocal/StreetCLIP", 
                 freeze_backbone=True, dropout=DROPOUT):
        super().__init__()
        
        print(f"Loading StreetCLIP from: {model_name}")
        
        # Load StreetCLIP vision encoder
        try:
            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
            print("✓ Successfully loaded StreetCLIP pretrained weights")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
            print("Loading with random initialization...")
            config = CLIPVisionConfig.from_pretrained(model_name)
            self.vision_model = CLIPVisionModel(config)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print("✓ Froze StreetCLIP backbone")
        
        # Get embedding dimension (768 for CLIP-Base, 1024 for CLIP-Large)
        self.embed_dim = self.vision_model.config.hidden_size
        print(f"Embedding dimension: {self.embed_dim}")
        
        # Classification head - simpler than before due to better features
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, CLASSIFICATION_HEAD_DIM),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(CLASSIFICATION_HEAD_DIM, NUM_CITIES)
        )
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Preprocessed images [batch, 3, H, W]
        
        Returns:
            logits: Class predictions [batch, num_cities]
        """
        # Get vision embeddings
        outputs = self.vision_model(pixel_values=pixel_values)
        
        # Use pooled output (CLS token)
        pooled_output = outputs.pooler_output  # [batch, embed_dim]
        
        # Classify
        logits = self.classifier(pooled_output)
        
        return logits
