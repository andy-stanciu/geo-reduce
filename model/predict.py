import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

from vit import ViTGeoRegressor
from load_data import OpenGuessrDataset, get_transforms
from device import get_device

# Initialize model
device = get_device()

# Option 1: Standard approach (resize to 224x224)
model = ViTGeoRegressor('vit_base_patch16_224', pretrained=True)
image_size = 224

model = model.to(device)
model.eval()

# Prepare dataset
transform = get_transforms(image_size=image_size)
dataset = OpenGuessrDataset(
    data_dir='./data',
    transform=transform,
    normalize_coords=True
)

print("\nCity distribution:")
city_dist = dataset.get_city_distribution()
for city, count in sorted(city_dist.items(), key=lambda x: x[1], reverse=True):
    print(f"  {city}: {count} images")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Inference
predictions = []
with torch.no_grad():
    for images, coords, metadata in tqdm(dataloader, desc="Predicting", unit="batch"):
        images = images.to(device)
        outputs = model(images)  # [batch_size, 2]
        
        # Vectorized denormalization
        lats_norm = outputs[:, 0].cpu().numpy()  # All latitudes
        longs_norm = outputs[:, 1].cpu().numpy()  # All longitudes
        
        lats = lats_norm * 90.0
        longs = longs_norm * 180.0
        
        # Batch append
        for i in range(len(lats)):
            predictions.append({
                'image': metadata['filename'][i],
                'city': metadata['city'][i],
                'latitude': lats[i],
                'longitude': longs[i],
                'true_latitude': metadata['raw_lat'][i].item(),
                'true_longitude': metadata['raw_long'][i].item()
            })

df = pd.DataFrame(predictions)
df.to_csv('openguessr_predictions.csv', index=False)

