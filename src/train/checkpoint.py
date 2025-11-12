import torch
import os


def save_checkpoint(model, optimizer, scheduler, epoch, 
                   train_loss, val_loss,
                   train_top1_acc, val_top1_acc,
                   train_top5_acc, val_top5_acc,
                   train_distance, val_distance,
                   save_dir, is_best=False):
    """
    Save model checkpoint with classification metrics
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        train_top1_acc: Training top-1 accuracy
        val_top1_acc: Validation top-1 accuracy
        train_top5_acc: Training top-5 accuracy
        val_top5_acc: Validation top-5 accuracy
        train_distance: Training distance error (km)
        val_distance: Validation distance error (km)
        save_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        
        # Loss metrics
        'train_loss': train_loss,
        'val_loss': val_loss,
        
        # Accuracy metrics
        'train_top1_acc': train_top1_acc,
        'val_top1_acc': val_top1_acc,
        'train_top5_acc': train_top5_acc,
        'val_top5_acc': val_top5_acc,
        
        # Distance metrics
        'train_distance': train_distance,
        'val_distance': val_distance,
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint (based on top-1 accuracy)
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ New best model saved (Top-1 acc: {val_top1_acc*100:.1f}%, Dist: {val_distance:.0f}km)")
