import torch
from datetime import datetime
import os
from torch.utils.data import DataLoader
from vit import ViTGeoRegressor
from device import get_device
from tqdm import tqdm
import pandas as pd
from device import get_device
from dataloader import OpenGuessrDataset, get_transforms, denormalize_coordinates
from train import haversine_distance


def load_model_for_inference(checkpoint_path, model_name='vit_base_patch16_224'):
    """
    Load a trained model from checkpoint for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., './checkpoints/run_XXX/checkpoint_best.pth')
        model_name: ViT model architecture name
        device: Device to load model on (default: auto-detect)
    
    Returns:
        model: Loaded model in eval mode, ready for inference
        checkpoint_info: Dictionary with training metadata (epoch, losses, etc.)
    """
    device = get_device()
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model architecture (must match training config)
    model = ViTGeoRegressor(
        model_name=model_name,
        pretrained=False,  # Don't load ImageNet weights, we have trained weights
        freeze_backbone=False  # Not relevant for inference
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Extract checkpoint info
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
    }
    
    print(f"✓ Model loaded successfully")
    print(f"  Trained for {checkpoint_info['epoch']} epochs")
    print(f"  Final train loss: {checkpoint_info['train_loss']:.4f}")
    print(f"  Final val loss: {checkpoint_info['val_loss']:.4f}\n")
    
    return model, checkpoint_info


def predict_batch(model, images, device=None):
    """
    Run inference on a batch of images.
    
    Args:
        model: Trained model in eval mode
        images: Tensor of shape [batch, 3, 224, 224]
        device: Device to run inference on
    
    Returns:
        predictions: Tensor of shape [batch, 2] with [lat, lon] pairs
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Ensure model is in eval mode
    model.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        images = images.to(device)
        predictions = model(images, clip_output=True)
    
    return predictions


def predict():    
    # Load trained model
    model, _ = load_model_for_inference(
        checkpoint_path='./checkpoints/run_20251111_125620/checkpoint_best.pth',
        model_name='vit_base_patch16_224'
    )
    
    device = next(model.parameters()).device
    
    # Load dataset
    dataset = OpenGuessrDataset(
        data_dir='./data',
        transform=get_transforms(),
        normalize_coords=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=1
    )
    
    # Run inference
    predictions = []
    
    print("Running inference...")
    with torch.no_grad():  # Disable gradient tracking
        for images, coords, metadata in tqdm(dataloader, desc="Predicting"):
            # Predict
            outputs = predict_batch(model, images, device)
            
            # Process batch
            batch_size = outputs.shape[0]
            for i in range(batch_size):
                lat_norm = outputs[i, 0].item()
                lon_norm = outputs[i, 1].item()
                lat, lon = denormalize_coordinates(lat_norm, lon_norm)
                
                predictions.append({
                    'image': metadata['filename'][i],
                    'city': metadata['city'][i],
                    'predicted_lat': lat,
                    'predicted_lon': lon,
                    'true_lat': metadata['raw_lat'][i].item(),
                    'true_lon': metadata['raw_long'][i].item()
                })
    
    # Save results
    df = pd.DataFrame(predictions)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('predictions', exist_ok=True)
    df.to_csv(f'predictions/predictions_{timestamp}.csv', index=False)
    
    print(f"\n✓ Predictions saved to predictions.csv")
    print(f"✓ Total predictions: {len(predictions)}")
    
    # Calculate average error
    
    errors = []
    for pred in predictions:
        error = haversine_distance(
            pred['predicted_lat'], pred['predicted_lon'],
            pred['true_lat'], pred['true_lon']
        )
        errors.append(error)
    
    avg_error = sum(errors) / len(errors)
    print(f"✓ Average distance error: {avg_error:.1f} km\n")
    
    return predictions


if __name__ == '__main__':
    predict()
