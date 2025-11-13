import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig

from data.city_mapping import NUM_CITIES
from constants import *


class StreetCLIPCityClassifier(nn.Module):
    def __init__(self, model_name="geolocal/StreetCLIP", 
                 freeze_backbone=True, 
                 unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS,
                 dropout=DROPOUT,
                 classifier_hidden_dim=CLASSIFICATION_HEAD_DIM):
        super().__init__()
        
        # Load StreetCLIP vision encoder
        try:
            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
            config = CLIPVisionConfig.from_pretrained(model_name)
            self.vision_model = CLIPVisionModel(config)
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            
            # Unfreeze last N transformer blocks
            if unfreeze_last_n_blocks > 0:
                num_blocks = len(self.vision_model.vision_model.encoder.layers)
                for i in range(num_blocks - unfreeze_last_n_blocks, num_blocks):
                    for param in self.vision_model.vision_model.encoder.layers[i].parameters():
                        param.requires_grad = True
                
                # print(f"✓ Unfroze last {unfreeze_last_n_blocks}/{num_blocks} transformer blocks")
        
        # Get embedding dimension
        self.embed_dim = self.vision_model.config.hidden_size
        
        # Larger classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, classifier_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, NUM_CITIES)
        )
    
    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
