import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
import math

from dataloader import load_classification_dataset
from vit import ViTCityClassifier
from device import get_device
from checkpoint import save_checkpoint
from constants import *
from city_mapping import *


def haversine_distance(lat1, long1, lat2, long2):
    """
    Calculate great-circle distance between two points on Earth (in km)
    using Haversine formula
    """
    # Convert to radians
    lat1, long1, lat2, long2 = map(math.radians, [lat1, long1, lat2, long2])
    
    dlat = lat2 - lat1
    dlong = long2 - long1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlong/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def calculate_metrics(outputs, labels, metadata):
    """
    Calculate accuracy and geographic distance metrics
    
    Args:
        outputs: Model logits [batch, 50]
        labels: True city labels [batch]
        meta Dict with true lat/lon
    
    Returns:
        top1_acc: Top-1 accuracy
        top5_acc: Top-5 accuracy
        avg_distance: Average distance error in km
    """
    batch_size = outputs.size(0)
    
    # Top-1 accuracy
    _, pred_labels = torch.max(outputs, dim=1)
    top1_correct = (pred_labels == labels).sum().item()
    top1_acc = top1_correct / batch_size
    
    # Top-5 accuracy
    _, top5_pred = torch.topk(outputs, k=min(5, NUM_CITIES), dim=1)
    top5_correct = sum([labels[i] in top5_pred[i] for i in range(batch_size)])
    top5_acc = top5_correct / batch_size
    
    # Geographic distance error
    total_distance = 0.0
    for i in range(batch_size):
        pred_city_idx = pred_labels[i].item()
        pred_lat, pred_lon = IDX_TO_COORDS[pred_city_idx]
        
        true_lat = metadata['true_lat'][i]
        true_lon = metadata['true_lon'][i]
        
        if true_lat is not None and true_lon is not None:
            distance = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
            total_distance += distance
    
    avg_distance = total_distance / batch_size
    
    return top1_acc, top5_acc, avg_distance


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_distance = 0.0
    running_top1_acc = 0.0
    running_top5_acc = 0.0
    running_distance = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]', 
                leave=False, dynamic_ncols=True)
    
    for images, labels, metadata in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        
        with torch.no_grad():
            top1_acc, top5_acc, avg_dist = calculate_metrics(outputs, labels, metadata)
            running_top1_acc += top1_acc * images.size(0)
            running_top5_acc += top5_acc * images.size(0)
            running_distance += avg_dist * images.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{top1_acc*100:.1f}%',
            'dist': f'{avg_dist:.0f}km'
        })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_top1_acc = running_top1_acc / len(dataloader.dataset)
    epoch_top5_acc = running_top5_acc / len(dataloader.dataset)
    epoch_distance = running_distance / len(dataloader.dataset)
    
    return epoch_loss, epoch_top1_acc, epoch_top5_acc, epoch_distance


def validate(model, dataloader, criterion, device, epoch, total_epochs):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_top1_acc = 0.0
    running_top5_acc = 0.0
    running_distance = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Val]  ', 
                leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        for images, labels, metadata in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            batch_loss = loss.item()
            running_loss += batch_loss * images.size(0)
            
            top1_acc, top5_acc, avg_dist = calculate_metrics(outputs, labels, metadata)
            running_top1_acc += top1_acc * images.size(0)
            running_top5_acc += top5_acc * images.size(0)
            running_distance += avg_dist * images.size(0)
            
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{top1_acc*100:.1f}%',
                'dist': f'{avg_dist:.0f}km'
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_top1_acc = running_top1_acc / len(dataloader.dataset)
    epoch_top5_acc = running_top5_acc / len(dataloader.dataset)
    epoch_distance = running_distance / len(dataloader.dataset)
    
    return epoch_loss, epoch_top1_acc, epoch_top5_acc, epoch_distance


def main(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'classifier_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    device = get_device()
    
    # Load dataset
    train_loader, val_loader, _ = load_classification_dataset(
        data_dir=args.data_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Initialize model
    print(f"Loading model: {args.model_name}")
    model = ViTCityClassifier(
        model_name=args.model_name,
        pretrained=True,
        freeze_backbone=True,
        dropout=args.dropout
    )
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}\n")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    
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
    
    best_val_acc = 0.0
    best_val_distance = float('inf')
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_top1_acc': [], 'val_top1_acc': [],
        'train_top5_acc': [], 'val_top5_acc': [],
        'train_distance': [], 'val_distance': []
    }

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc='Training Progress',
                      position=0, dynamic_ncols=True)

    for epoch in epoch_pbar:
        # Train
        train_loss, train_top1, train_top5, train_dist = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # Validate
        val_loss, val_top1, val_top5, val_dist = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_top1_acc'].append(train_top1)
        history['val_top1_acc'].append(val_top1)
        history['train_top5_acc'].append(train_top5)
        history['val_top5_acc'].append(val_top5)
        history['train_distance'].append(train_dist)
        history['val_distance'].append(val_dist)
        
        # Update LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'train_acc': f'{train_top1*100:.1f}%',
            'val_acc': f'{val_top1*100:.1f}%',
            'val_dist': f'{val_dist:.0f}km',
            'lr': f'{current_lr:.2e}'
        })
        
        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train: Loss={train_loss:.4f}, Top-1={train_top1*100:.1f}%, Top-5={train_top5*100:.1f}%, Dist={train_dist:.0f}km")
        print(f"  Val:   Loss={val_loss:.4f}, Top-1={val_top1*100:.1f}%, Top-5={val_top5*100:.1f}%, Dist={val_dist:.0f}km")
        print(f"  LR: {current_lr:.2e}\n")
        
        # Check if this is the best model (based on top-1 accuracy)
        is_best = val_top1 > best_val_acc
        if is_best:
            best_val_acc = val_top1
            best_val_distance = val_dist
        
        # Save checkpoint every N epochs or if best
        if epoch % args.save_freq == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_top1_acc=train_top1,
                val_top1_acc=val_top1,
                train_top5_acc=train_top5,
                val_top5_acc=val_top5,
                train_distance=train_dist,
                val_distance=val_dist,
                save_dir=save_dir,
                is_best=is_best
            )
    
    # Save final metrics
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Val Top-1 Accuracy: {best_val_acc*100:.1f}%")
    print(f"Best Val Distance Error: {best_val_distance:.0f} km")
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
                       help='Train split ratio')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Val split ratio')
    
    # Model
    parser.add_argument('--model-name', type=str, default='vit_base_patch16_224',
                       help='ViT model name from timm')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5,
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
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)
