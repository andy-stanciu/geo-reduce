import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import glob
import re
import os

# lat/long range for continental USA
LAT_MIN, LAT_MAX = 24.5, 49.4
LONG_MIN, LONG_MAX = -125.0, -66.9

class OpenGuessrDataset(Dataset):
    """Dataset for OpenGuessr Street View images with nested city directories"""
    
    def __init__(self, data_dir, transform=None, normalize_coords=True):
        """
        Args:
            data_dir: Root directory containing city subdirectories (e.g., './data')
            transform: Optional image transforms
            normalize_coords: If True, normalize lat/long to [-1, 1]
        """
        self.data_dir = data_dir
        self.transform = transform
        self.normalize_coords = normalize_coords
        
        # Regex pattern to extract (lat, long) from filename
        # Matches: "CityName_ (37.712119, -97.325257).jpg"
        self.coord_pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'
        
        # Collect all image paths from nested city directories
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} images from {len(self.get_cities())} cities")
    
    def _load_samples(self):
        """Recursively load all images from city subdirectories"""
        city_dirs = [d for d in glob.glob(os.path.join(self.data_dir, '*')) 
                     if os.path.isdir(d)]
        
        for city_dir in city_dirs:
            city_name = os.path.basename(city_dir)
            
            # Get all jpg/png images in this city directory
            image_files = glob.glob(os.path.join(city_dir, '*.jpg')) + \
                         glob.glob(os.path.join(city_dir, '*.png'))
            
            for img_path in image_files:
                filename = os.path.basename(img_path)
                
                # Extract lat/long from filename using regex
                match = re.search(self.coord_pattern, filename)
                if match:
                    lat = float(match.group(1))
                    long = float(match.group(2))
                    
                    self.samples.append({
                        'path': img_path,
                        'city': city_name,
                        'lat': lat,
                        'long': long,
                        'filename': filename
                    })
                else:
                    print(f"Warning: Could not extract coordinates from {filename}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            coords: Tensor of [lat, long] (normalized if normalize_coords=True)
            meta Dictionary with additional info
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get coordinates
        lat, long = sample['lat'], sample['long']
        
        if self.normalize_coords:
            lat_norm = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
            long_norm = (long - LONG_MIN) / (LONG_MAX - LONG_MIN)
            coords = torch.tensor([lat_norm, long_norm], dtype=torch.float32)
        else:
            coords = torch.tensor([lat, long], dtype=torch.float32)
        
        # Metadata for debugging/analysis
        metadata = {
            'city': sample['city'],
            'filename': sample['filename'],
            'raw_lat': sample['lat'],
            'raw_long': sample['long']
        }
        
        return image, coords, metadata
    
    def get_cities(self):
        """Return list of unique cities in dataset"""
        return list(set([s['city'] for s in self.samples]))
    
    def get_city_distribution(self):
        """Return dictionary of image counts per city"""
        from collections import Counter
        return dict(Counter([s['city'] for s in self.samples]))

def denormalize_coordinates(lat_norm, long_norm):
    """Convert normalized [0, 1] back to USA lat/long"""
    lat = lat_norm * (LAT_MAX - LAT_MIN) + LAT_MIN
    long = long_norm * (LONG_MAX - LONG_MIN) + LONG_MIN
    
    # Clipping ensures valid bounds
    lat = max(LAT_MIN, min(LAT_MAX, lat))
    long = max(LONG_MIN, min(LONG_MAX, long))
    
    return lat, long


def get_transforms(image_size=224):
    """
    Get preprocessing transforms
    
    Options:
    1. Resize 500x500 -> 224x224 (quality loss but simple)
    2. Center crop 500x500 -> 224x224 (preserves quality, loses edges)
    3. Resize to 384 or 448 with position embedding interpolation

    For now, going with option 1.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), 
                         interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
