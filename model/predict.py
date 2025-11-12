"""
predict.py - Inference with city classification model
"""
import torch
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd

from vit import ViTCityClassifier
from device import get_device
from dataloader import load_classification_dataset
from city_mapping import IDX_TO_CITY, IDX_TO_COORDS, CITY_NAMES
import math


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def load_classifier_for_inference(checkpoint_path, model_name='vit_base_patch16_224'):
    """
    Load trained classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_name: ViT architecture name
    
    Returns:
        model: Loaded model in eval mode
        checkpoint_info: Training metadata
    """
    device = get_device()
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = ViTCityClassifier(
        model_name=model_name,
        pretrained=False,
        freeze_backbone=True,
        dropout=0.3
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Extract checkpoint info
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint.get('train_loss', 'N/A'),
        'val_loss': checkpoint.get('val_loss', 'N/A'),
        'val_top1_acc': checkpoint.get('val_top1_acc', 'N/A'),
        'val_top5_acc': checkpoint.get('val_top5_acc', 'N/A'),
        'val_distance': checkpoint.get('val_distance', 'N/A'),
    }
    
    print(f"✓ Model loaded successfully")
    print(f"  Trained for {checkpoint_info['epoch']} epochs")
    
    if checkpoint_info['val_top1_acc'] != 'N/A':
        print(f"  Val Top-1 Accuracy: {checkpoint_info['val_top1_acc']*100:.1f}%")
        print(f"  Val Top-5 Accuracy: {checkpoint_info['val_top5_acc']*100:.1f}%")
        print(f"  Val Distance Error: {checkpoint_info['val_distance']:.0f} km\n")
    else:
        print(f"  Train Loss: {checkpoint_info['train_loss']:.4f}")
        print(f"  Val Loss: {checkpoint_info['val_loss']:.4f}\n")
    
    return model, checkpoint_info


def predict_batch(model, images, device, top_k=5):
    """
    Run inference and get top-k predictions.
    
    Args:
        model: Trained classifier
        images: Image tensor [batch, 3, 224, 224]
        device: Device to run on
        top_k: Number of top predictions to return
    
    Returns:
        top1_pred: Top-1 predicted class indices [batch]
        topk_pred: Top-k predicted class indices [batch, k]
        topk_probs: Top-k probabilities [batch, k]
    """
    model.eval()
    
    with torch.no_grad():
        images = images.to(device)
        logits = model(images)  # [batch, 50]
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Top-1 prediction
        top1_pred = torch.argmax(logits, dim=1)  # [batch]
        
        # Top-k predictions
        topk_probs, topk_pred = torch.topk(probs, k=min(top_k, len(CITY_NAMES)), dim=1)
    
    return top1_pred, topk_pred, topk_probs


def predict(checkpoint_path='./checkpoints/best_classifier.pth', 
           data_dir='./data',
           batch_size=32,
           use_test_set=True):
    """
    Run inference on dataset.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to data directory
        batch_size: Batch size for inference
        use_test_set: If True, use test set; if False, use validation set
    """
    # Load model
    model, info = load_classifier_for_inference(
        checkpoint_path=checkpoint_path,
        model_name='vit_base_patch16_224'
    )
    
    device = next(model.parameters()).device
    
    # Load dataset
    print("Loading dataset...")
    _, val_loader, test_loader = load_classification_dataset(
        data_dir=data_dir,
        train_split=0.8,
        val_split=0.1,
        batch_size=batch_size,
        seed=42
    )
    
    dataloader = test_loader if use_test_set else val_loader
    dataset_name = "Test" if use_test_set else "Validation"
    
    # Run inference
    predictions = []
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    print(f"\nRunning inference on {dataset_name} set...")
    
    with torch.no_grad():
        for images, labels, metadata in tqdm(dataloader, desc="Predicting"):
            # Get predictions
            top1_pred, top5_pred, top5_probs = predict_batch(model, images, device, top_k=5)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Process each sample in batch
            for i in range(batch_size):
                # True information
                true_city_idx = labels[i].item()
                true_city = IDX_TO_CITY[true_city_idx]
                true_lat = metadata['true_lat'][i]
                true_lon = metadata['true_lon'][i]
                
                # Top-1 prediction
                pred_city_idx = top1_pred[i].item()
                pred_city = IDX_TO_CITY[pred_city_idx]
                pred_lat, pred_lon = IDX_TO_COORDS[pred_city_idx]
                
                # Top-5 predictions
                top5_cities = [IDX_TO_CITY[idx.item()] for idx in top5_pred[i]]
                top5_probs_list = [prob.item() for prob in top5_probs[i]]
                
                # Calculate metrics
                is_top1_correct = (pred_city_idx == true_city_idx)
                is_top5_correct = (true_city_idx in top5_pred[i])
                
                if is_top1_correct:
                    correct_top1 += 1
                if is_top5_correct:
                    correct_top5 += 1
                
                # Calculate distance error
                if true_lat is not None and true_lon is not None:
                    distance_error = haversine_distance(
                        pred_lat, pred_lon, true_lat, true_lon
                    )
                else:
                    distance_error = None
                
                # Store prediction
                predictions.append({
                    'filename': metadata['filename'][i],
                    'true_city': true_city,
                    'predicted_city': pred_city,
                    'is_correct': is_top1_correct,
                    'predicted_lat': pred_lat,
                    'predicted_lon': pred_lon,
                    'true_lat': true_lat if true_lat is not None else 'N/A',
                    'true_lon': true_lon if true_lon is not None else 'N/A',
                    'distance_error_km': distance_error if distance_error is not None else 'N/A',
                    'confidence': top5_probs_list[0],
                    'top5_cities': ', '.join(top5_cities),
                    'top5_confidences': ', '.join([f'{p:.3f}' for p in top5_probs_list])
                })
    
    # Calculate final metrics
    top1_accuracy = correct_top1 / total_samples
    top5_accuracy = correct_top5 / total_samples
    
    # Calculate average distance error
    valid_distances = [p['distance_error_km'] for p in predictions 
                      if p['distance_error_km'] != 'N/A']
    avg_distance = sum(valid_distances) / len(valid_distances) if valid_distances else 0
    
    # Save results
    df = pd.DataFrame(predictions)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('predictions', exist_ok=True)
    output_file = f'predictions/classifier_predictions_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Inference Results on {dataset_name} Set")
    print(f"{'='*70}")
    print(f"Total samples: {total_samples}")
    print(f"Top-1 Accuracy: {top1_accuracy*100:.2f}% ({correct_top1}/{total_samples})")
    print(f"Top-5 Accuracy: {top5_accuracy*100:.2f}% ({correct_top5}/{total_samples})")
    print(f"Average Distance Error: {avg_distance:.1f} km")
    print(f"\nPredictions saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Show some example predictions
    print("Sample Predictions:")
    print("-" * 70)
    for i, pred in enumerate(predictions[:5]):
        status = "✓" if pred['is_correct'] else "✗"
        print(f"{status} True: {pred['true_city']:20s} | Pred: {pred['predicted_city']:20s} | "
              f"Conf: {pred['confidence']:.1%} | Dist: {pred['distance_error_km']:.0f}km" 
              if pred['distance_error_km'] != 'N/A' else 
              f"{status} True: {pred['true_city']:20s} | Pred: {pred['predicted_city']:20s} | "
              f"Conf: {pred['confidence']:.1%}")
    
    return predictions


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with city classifier')
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints/best_classifier.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--use-val', action='store_true',
                       help='Use validation set instead of test set')
    
    args = parser.parse_args()
    
    predict(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_test_set=not args.use_val
    )
