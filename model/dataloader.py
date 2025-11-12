import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from pathlib import Path
import glob
import re
import os
from constants import *
from city_mapping import *


def load_classification_dataset(data_dir, train_split=0.8, val_split=0.1,
                                batch_size=32, seed=42):
    """
    Load dataset and create train/val/test splits
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Load full dataset with augmentation for training
    full_dataset = OpenGuessrClassificationDataset(
        data_dir=data_dir,
        transform=None  # We'll apply transforms separately
    )
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_transforms(augment=False)
    val_dataset.dataset.transform = get_transforms(augment=False)
    test_dataset.dataset.transform = get_transforms(augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images\n")
    
    return train_loader, val_loader, test_loader


class OpenGuessrClassificationDataset(Dataset):
    """Dataset for city classification"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Root directory containing city subdirectories
            transform: Image transforms
        """
        self.data_dir = data_dir
        self.transform = transform
        self.coord_pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'
        self.samples = []
        self._load_samples()
        
        # Print class distribution
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load all images and assign city labels"""
        city_dirs = [d for d in glob.glob(os.path.join(self.data_dir, '*')) 
                     if os.path.isdir(d)]
        
        for city_dir in city_dirs:
            city_name = os.path.basename(city_dir)
            
            # Map directory name to city label
            if city_name not in CITY_TO_IDX:
                print(f"⚠️ Warning: '{city_name}' not in city mapping, skipping")
                continue
            
            city_label = CITY_TO_IDX[city_name]
            
            # Get all images
            image_files = glob.glob(os.path.join(city_dir, '*.jpg')) + \
                         glob.glob(os.path.join(city_dir, '*.png'))
            
            for img_path in image_files:
                filename = os.path.basename(img_path)
                
                # Extract true coordinates (for evaluation)
                match = re.search(self.coord_pattern, filename)
                if match:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                else:
                    lat, lon = None, None
                
                self.samples.append({
                    'path': img_path,
                    'city': city_name,
                    'city_label': city_label,
                    'true_lat': lat,
                    'true_lon': lon,
                    'filename': filename
                })
        
        print(f"Loaded {len(self.samples)} images from {len(set(s['city_label'] for s in self.samples))} cities")
    
    def _print_class_distribution(self):
        """Print how many images per city"""
        from collections import Counter
        city_counts = Counter([s['city_label'] for s in self.samples])
        
        print(f"\nClass distribution:")
        print(f"  Min images per city: {min(city_counts.values())}")
        print(f"  Max images per city: {max(city_counts.values())}")
        print(f"  Avg images per city: {sum(city_counts.values()) / len(city_counts):.1f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            label: City class label (0-49)
            metadata: Dictionary with additional info
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # City label (integer 0-49)
        label = torch.tensor(sample['city_label'], dtype=torch.long)
        
        # Metadata
        metadata = {
            'city': sample['city'],
            'city_label': sample['city_label'],
            'filename': sample['filename'],
            'true_lat': sample['true_lat'],
            'true_lon': sample['true_lon']
        }
        
        return image, label, metadata
    

def denormalize_coordinates(lat_norm, long_norm):
    """Convert normalized [0, 1] back to USA lat/long"""
    lat = lat_norm * (LAT_MAX - LAT_MIN) + LAT_MIN
    long = long_norm * (LONG_MAX - LONG_MIN) + LONG_MIN
    
    # Clipping ensures valid bounds
    lat = max(LAT_MIN, min(LAT_MAX, lat))
    long = max(LONG_MIN, min(LONG_MAX, long))
    
    return lat, long


def get_transforms(augment=False, image_size=224):
    """Get transforms with optional augmentation"""
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size),
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
