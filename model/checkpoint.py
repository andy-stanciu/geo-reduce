import torch
import os

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                   save_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ New best model saved (val_loss: {val_loss:.4f})")

