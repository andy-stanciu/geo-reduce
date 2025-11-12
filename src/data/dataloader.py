"""
dataloader.py - Dataset and transforms for StreetCLIP
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from collections import Counter
import glob
import re
import os

from constants import *
from data.city_mapping import *


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


def get_clip_transforms(augment=False, image_size=CLIP_IMG_SIZE):
    """
    Get transforms for CLIP/StreetCLIP.
    
    CLIP expects images normalized with specific mean/std.
    """
    
    # optional data augmentation in an effort to reduce overfitting
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


def load_classification_dataset(data_dir, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                                batch_size=BATCH_SIZE, seed=SEED):
    """
    Load dataset with appropriate transforms.
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
    
    train_dataset.dataset.transform = get_clip_transforms(augment=False)
    val_dataset.dataset.transform = get_clip_transforms(augment=False)
    test_dataset.dataset.transform = get_clip_transforms(augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images\n")
    
    return train_loader, val_loader, test_loader
