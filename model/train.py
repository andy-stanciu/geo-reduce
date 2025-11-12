"""
train.py - Train StreetCLIP city classifier with visualization
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
import math
import matplotlib.pyplot as plt

from dataloader import load_classification_dataset
from vit import StreetCLIPCityClassifier
from device import get_device
from checkpoint import save_checkpoint
from constants import *
from city_mapping import *


def haversine_distance(lat1, long1, lat2, long2):
    """Calculate great-circle distance between two points on Earth (in km)"""
    lat1, long1, lat2, long2 = map(math.radians, [lat1, long1, lat2, long2])
    dlat = lat2 - lat1
    dlong = long2 - long1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlong/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def calculate_metrics(outputs, labels, metadata):
    """Calculate accuracy and geographic distance metrics"""
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
    running_top1_acc = 0.0
    running_top5_acc = 0.0
    running_distance = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]', 
                leave=False, dynamic_ncols=True)
    
    for images, labels, metadata in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        
        with torch.no_grad():
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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
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


def plot_training_curves(history, save_dir):
    """
    Plot training and validation curves for loss, accuracy, and distance.
    
    Args:
        history: Dictionary with training metrics
        save_dir: Directory to save plots
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top-1 Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, [acc*100 for acc in history['train_top1_acc']], 
             'b-o', label='Train Top-1 Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, [acc*100 for acc in history['val_top1_acc']], 
             'r-s', label='Val Top-1 Acc', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax2.set_title('Top-1 Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Top-5 Accuracy
    ax3 = axes[1, 0]
    ax3.plot(epochs, [acc*100 for acc in history['train_top5_acc']], 
             'b-o', label='Train Top-5 Acc', linewidth=2, markersize=4)
    ax3.plot(epochs, [acc*100 for acc in history['val_top5_acc']], 
             'r-s', label='Val Top-5 Acc', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Top-5 Accuracy (%)', fontsize=12)
    ax3.set_title('Top-5 Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance Error
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['train_distance'], 
             'b-o', label='Train Dist Error', linewidth=2, markersize=4)
    ax4.plot(epochs, history['val_distance'], 
             'r-s', label='Val Dist Error', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Distance Error (km)', fontsize=12)
    ax4.set_title('Geographic Distance Error', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {plot_path}")


def main(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_type = "streetclip"
    save_dir = os.path.join(args.save_dir, f'{model_type}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    device = get_device()
    
    print(f"{'='*70}")
    print(f"OpenGuessr City Classifier Training")
    print(f"Model: StreetClip")
    print(f"{'='*70}\n")
    
    # Load dataset
    train_loader, val_loader, _ = load_classification_dataset(
        data_dir=args.data_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Initialize model
    model = StreetCLIPCityClassifier(
        model_name="geolocal/StreetCLIP",
        freeze_backbone=True,
        dropout=args.dropout
    )
    
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}\n")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
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
    
    # History tracking
    history = {
        'train_loss': [], 
        'val_loss': [],
        'train_top1_acc': [], 
        'val_top1_acc': [],
        'train_top5_acc': [], 
        'val_top5_acc': [],
        'train_distance': [], 
        'val_distance': []
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
        
        # Store metrics
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
        
        # Check if best model
        is_best = val_top1 > best_val_acc
        if is_best:
            best_val_acc = val_top1
            best_val_distance = val_dist
        
        # Save checkpoint
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
        
        # Save history after each epoch (so you don't lose progress)
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
    
    # Generate final plots
    print("\nGenerating training curve plots...")
    plot_training_curves(history, save_dir)
    
    # Save final summary
    summary = {
        'best_val_top1_acc': best_val_acc,
        'best_val_distance': best_val_distance,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_train_top1_acc': history['train_top1_acc'][-1],
        'final_val_top1_acc': history['val_top1_acc'][-1],
        'total_epochs': args.epochs,
        'model_type': model_type,
        'trainable_params': trainable_params,
        'total_params': total_params
    }
    
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Val Top-1 Accuracy: {best_val_acc*100:.1f}%")
    print(f"Best Val Distance Error: {best_val_distance:.0f} km")
    print(f"Final Train/Val Gap: {(history['train_top1_acc'][-1] - history['val_top1_acc'][-1])*100:.1f}%")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train city classifier')
    
    # Data
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--save-dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--train-split', type=float, default=TRAIN_SPLIT)
    parser.add_argument('--val-split', type=float, default=VAL_SPLIT)
    
    # Training
    parser.add_argument('--epochs', type=int, default=TRAIN_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--min-lr', type=float, default=MIN_LR)
    parser.add_argument('--weight-decay', type=float, default=ADAMW_WEIGHT_DECAY)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--seed', type=int, default=SEED)
    
    args = parser.parse_args()
    main(args)
