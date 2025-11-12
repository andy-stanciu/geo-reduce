"""
dataloader.py - Dataset and transforms for StreetCLIP
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import glob
import re
import os

from constants import *
from city_mapping import *


class OpenGuessrClassificationDataset(Dataset):
    """Dataset for city classification with StreetCLIP"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.coord_pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'
        self.samples = []
        self._load_samples()
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load all images and assign city labels"""
        city_dirs = [d for d in glob.glob(os.path.join(self.data_dir, '*')) 
                     if os.path.isdir(d)]
        
        for city_dir in city_dirs:
            city_name = os.path.basename(city_dir)
            
            if city_name not in CITY_TO_IDX:
                print(f"⚠️ Warning: '{city_name}' not in city mapping, skipping")
                continue
            
            city_label = CITY_TO_IDX[city_name]
            image_files = glob.glob(os.path.join(city_dir, '*.jpg')) + \
                         glob.glob(os.path.join(city_dir, '*.png'))
            
            for img_path in image_files:
                filename = os.path.basename(img_path)
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
        from collections import Counter
        city_counts = Counter([s['city_label'] for s in self.samples])
        print(f"\nClass distribution:")
        print(f"  Min images per city: {min(city_counts.values())}")
        print(f"  Max images per city: {max(city_counts.values())}")
        print(f"  Avg images per city: {sum(city_counts.values()) / len(city_counts):.1f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(sample['city_label'], dtype=torch.long)
        
        metadata = {
            'city': sample['city'],
            'city_label': sample['city_label'],
            'filename': sample['filename'],
            'true_lat': sample['true_lat'],
            'true_lon': sample['true_lon']
        }
        
        return image, label, metadata


def get_clip_transforms(augment=False, image_size=336):
    """
    Get transforms for CLIP/StreetCLIP.
    
    CLIP expects images normalized with specific mean/std.
    Note: StreetCLIP uses 336x336 images, but we'll use 224 for consistency
    """
    # CLIP normalization (different from ImageNet!)
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.15
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size),
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])


def load_classification_dataset(data_dir, train_split=0.8, val_split=0.1,
                                batch_size=32, seed=42, use_clip=True):
    """
    Load dataset with appropriate transforms.
    
    Args:
        use_clip: If True, use CLIP transforms; if False, use ImageNet transforms
    """
    full_dataset = OpenGuessrClassificationDataset(
        data_dir=data_dir,
        transform=None
    )
    
    train_size = int(train_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply transforms based on model type
    if use_clip:
        train_dataset.dataset.transform = get_clip_transforms(augment=True)
        val_dataset.dataset.transform = get_clip_transforms(augment=False)
        test_dataset.dataset.transform = get_clip_transforms(augment=False)
        print("Using CLIP transforms")
    else:
        # Old ImageNet transforms
        train_dataset.dataset.transform = get_transforms(augment=True)
        val_dataset.dataset.transform = get_transforms(augment=False)
        test_dataset.dataset.transform = get_transforms(augment=False)
        print("Using ImageNet transforms")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images\n")
    
    return train_loader, val_loader, test_loader


# Keep old function for backward compatibility
def get_transforms(augment=False, image_size=224):
    """ImageNet transforms (for non-CLIP models)"""
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                 saturation=0.4, hue=0.15),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def denormalize_coordinates(lat_norm, long_norm):
    """Convert normalized [0, 1] back to USA lat/long"""
    lat = lat_norm * (LAT_MAX - LAT_MIN) + LAT_MIN
    long = long_norm * (LONG_MAX - LONG_MIN) + LONG_MIN
    lat = max(LAT_MIN, min(LAT_MAX, lat))
    long = max(LONG_MIN, min(LONG_MAX, long))
    return lat, long
