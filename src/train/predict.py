"""
predict.py - Inference with StreetCLIP
"""
import torch
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd

from constants import *
from model.clip import StreetCLIPCityClassifier
from device import get_device
from data.dataloader import load_classification_dataset
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from data.city_mapping import IDX_TO_CITY, IDX_TO_COORDS, CITY_NAMES
import math


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def load_classifier_for_inference(checkpoint_path):
    """Load trained classifier from checkpoint"""
    device = get_device()
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = StreetCLIPCityClassifier(
        model_name="geolocal/StreetCLIP",
        freeze_backbone=True,
        dropout=DROPOUT
    )
    
    state_dict = checkpoint['model_state_dict']
    consume_prefix_in_state_dict_if_present(state_dict, 'module.')

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'val_top1_acc': checkpoint.get('val_top1_acc', 'N/A'),
        'val_top5_acc': checkpoint.get('val_top5_acc', 'N/A'),
        'val_distance': checkpoint.get('val_distance', 'N/A'),
    }
    
    print(f"✓ Model loaded successfully")
    print(f"  Trained for {checkpoint_info['epoch'] + 1} epochs")
    
    if checkpoint_info['val_top1_acc'] != 'N/A':
        print(f"  Val Top-1 Accuracy: {checkpoint_info['val_top1_acc']*100:.1f}%")
        print(f"  Val Top-5 Accuracy: {checkpoint_info['val_top5_acc']*100:.1f}%")
        print(f"  Val Distance Error: {checkpoint_info['val_distance']:.0f} km\n")
    
    return model, checkpoint_info


def predict_batch(model, images, device, top_k=5):
    """Run inference and get top-k predictions"""
    model.eval()
    
    with torch.no_grad():
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        top1_pred = torch.argmax(logits, dim=1)
        topk_probs, topk_pred = torch.topk(probs, k=min(top_k, len(CITY_NAMES)), dim=1)
    
    return top1_pred, topk_pred, topk_probs


def predict(checkpoint_path, data_dir='./data', batch_size=BATCH_SIZE,
           use_test_set=True):
    """Run inference on dataset"""
    
    model, _ = load_classifier_for_inference(
        checkpoint_path=checkpoint_path,
    )
    
    device = next(model.parameters()).device
    
    print("Loading dataset...")
    _, val_loader, test_loader = load_classification_dataset(
        data_dir=data_dir,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        batch_size=batch_size,
        seed=SEED
    )
    
    dataloader = test_loader if use_test_set else val_loader
    dataset_name = "Test" if use_test_set else "Validation"
    
    predictions = []
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    print(f"\nRunning inference on {dataset_name} set...")
    
    with torch.no_grad():
        for images, labels, metadata in tqdm(dataloader, desc="Predicting"):
            top1_pred, top5_pred, top5_probs = predict_batch(model, images, device, top_k=5)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            for i in range(batch_size):
                true_city_idx = labels[i].item()
                true_city = IDX_TO_CITY[true_city_idx]
                true_lat = metadata['true_lat'][i]
                true_lon = metadata['true_lon'][i]
                
                pred_city_idx = top1_pred[i].item()
                pred_city = IDX_TO_CITY[pred_city_idx]
                pred_lat, pred_lon = IDX_TO_COORDS[pred_city_idx]
                
                top5_cities = [IDX_TO_CITY[idx.item()] for idx in top5_pred[i]]
                top5_probs_list = [prob.item() for prob in top5_probs[i]]
                
                is_top1_correct = (pred_city_idx == true_city_idx)
                is_top5_correct = (true_city_idx in top5_pred[i])
                
                if is_top1_correct:
                    correct_top1 += 1
                if is_top5_correct:
                    correct_top5 += 1
                
                if true_lat is not None and true_lon is not None:
                    distance_error = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
                else:
                    distance_error = None
                
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
    
    top1_accuracy = correct_top1 / total_samples
    top5_accuracy = correct_top5 / total_samples
    
    valid_distances = [p['distance_error_km'] for p in predictions 
                      if p['distance_error_km'] != 'N/A']
    avg_distance = sum(valid_distances) / len(valid_distances) if valid_distances else 0
    
    df = pd.DataFrame(predictions)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('predictions', exist_ok=True)
    model_type = "streetclip"
    output_file = f'predictions/{model_type}_predictions_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"Inference Results on {dataset_name} Set")
    print(f"{'='*70}")
    print(f"Total samples: {total_samples}")
    print(f"Top-1 Accuracy: {top1_accuracy*100:.2f}% ({correct_top1}/{total_samples})")
    print(f"Top-5 Accuracy: {top5_accuracy*100:.2f}% ({correct_top5}/{total_samples})")
    print(f"Average Distance Error: {avg_distance:.1f} km")
    print(f"\nPredictions saved to: {output_file}")
    print(f"{'='*70}\n")
    
    print("Sample Predictions:")
    print("-" * 70)
    for pred in predictions[:5]:
        status = "✓" if pred['is_correct'] else "✗"
        if pred['distance_error_km'] != 'N/A':
            print(f"{status} True: {pred['true_city']:20s} | Pred: {pred['predicted_city']:20s} | "
                  f"Conf: {pred['confidence']:.1%} | Dist: {pred['distance_error_km']:.0f}km")
        else:
            print(f"{status} True: {pred['true_city']:20s} | Pred: {pred['predicted_city']:20s} | "
                  f"Conf: {pred['confidence']:.1%}")
    
    return predictions


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with city classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--use-val', action='store_true',
                       help='Use validation set instead of test set')
    
    args = parser.parse_args()
    
    predict(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_test_set=not args.use_val
    )
