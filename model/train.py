import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
import json
from datetime import datetime
import math

from dataloader import OpenGuessrDataset, denormalize_coordinates, get_transforms
from vit import ViTGeoRegressor
from device import get_device
from checkpoint import save_checkpoint


def haversine_distance(lat1, long1, lat2, long2):
    """
    Calculate great-circle distance between two points on Earth (in km)
    using Haversine formula
    """
    R = 6371  # Earth's radius in km
    
    # Convert to radians
    lat1, long1, lat2, long2 = map(math.radians, [lat1, long1, lat2, long2])
    
    dlat = lat2 - lat1
    dlong = long2 - long1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlong/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_distance = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]', 
                leave=False, dynamic_ncols=True)
    
    for images, coords, metadata in pbar:
        images = images.to(device)
        coords = coords.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, coords)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        
        # Calculate average distance error in km
        with torch.no_grad():
            batch_distance = 0.0
            for i in range(outputs.shape[0]):
                pred_lat, pred_lon = denormalize_coordinates(
                    outputs[i, 0].item(), outputs[i, 1].item()
                )
                true_lat = metadata['raw_lat'][i].item()
                true_lon = metadata['raw_long'][i].item()
                batch_distance += haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
            batch_distance /= outputs.shape[0]
            running_distance += batch_distance * images.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'dist': f'{batch_distance:.1f}km'
        })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_distance = running_distance / len(dataloader.dataset)
    
    return epoch_loss, epoch_distance


def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_distance = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Val]  ', 
                leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        for images, coords, metadata in pbar:
            images = images.to(device)
            coords = coords.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, coords)
            
            # Calculate metrics
            batch_loss = loss.item()
            running_loss += batch_loss * images.size(0)
            
            # Calculate average distance error
            batch_distance = 0.0
            for i in range(outputs.shape[0]):
                pred_lat, pred_lon = denormalize_coordinates(
                    outputs[i, 0].item(), outputs[i, 1].item()
                )
                true_lat = metadata['raw_lat'][i].item()
                true_lon = metadata['raw_long'][i].item()
                batch_distance += haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
            batch_distance /= outputs.shape[0]
            running_distance += batch_distance * images.size(0)
            
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'dist': f'{batch_distance:.1f}km'
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_distance = running_distance / len(dataloader.dataset)
    
    return epoch_loss, epoch_distance


def plot_losses(train_losses, val_losses, train_distances, val_distances, save_dir):
    """Plot and save training curves"""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot MSE Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot Distance Error
    ax2.plot(epochs, train_distances, 'b-o', label='Train Dist Error', linewidth=2, markersize=4)
    ax2.plot(epochs, val_distances, 'r-s', label='Val Dist Error', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Distance Error (km)', fontsize=12)
    ax2.set_title('Prediction Distance Error', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {os.path.join(save_dir, 'training_curves.png')}")


def main(args):
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Get device
    device = get_device()
    
    print(f"{'='*70}")
    print(f"OpenGuessr ViT Training")
    print(f"{'='*70}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {device}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"{'='*70}\n")
    
    # Load full dataset
    print("Loading dataset...")
    full_dataset = OpenGuessrDataset(
        data_dir=args.data_dir,
        transform=get_transforms(),
        normalize_coords=True
    )
    
    # Split into train/val
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # pin_memory=True
    )
    
    # Initialize model
    print(f"Loading model: {args.model_name}")
    model = ViTGeoRegressor(
        model_name=args.model_name,
        pretrained=True,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}\n")
    
    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...\n")
    
    train_losses = []
    val_losses = []
    train_distances = []
    val_distances = []
    best_val_loss = float('inf')
    
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc='Training Progress', 
                      position=0, dynamic_ncols=True)
    
    for epoch in epoch_pbar:
        # Train
        train_loss, train_dist = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        train_losses.append(train_loss)
        train_distances.append(train_dist)
        
        # Validate
        val_loss, val_dist = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )
        val_losses.append(val_loss)
        val_distances.append(val_dist)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update main progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_dist': f'{val_dist:.1f}km',
            'lr': f'{current_lr:.2e}'
        })
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Dist: {train_dist:.1f} km")
        print(f"  Val Loss:   {val_loss:.4f} | Val Dist:   {val_dist:.1f} km")
        print(f"  LR: {current_lr:.2e}\n")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % args.save_freq == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                train_loss, val_loss, save_dir, is_best
            )
    
    # Plot and save training curves
    print("\nGenerating training curves...")
    plot_losses(train_losses, val_losses, train_distances, val_distances, save_dir)
    
    # Save final metrics
    metrics = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': best_val_loss,
        'final_train_dist': train_distances[-1],
        'final_val_dist': val_distances[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_distances': train_distances,
        'val_distances': val_distances,
    }
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Final Val Distance Error: {val_distances[-1]:.1f} km")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ViT for OpenGuessr')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Train/val split ratio')
    
    # Model
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224',
                       help='ViT model name from timm')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze ViT backbone (only train head)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                       help='Minimum learning rate for cosine annealing')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # System
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)
