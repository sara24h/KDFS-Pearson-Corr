#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import numpy as np
from tqdm import tqdm
import pandas as pd

# Import your dataset module
from data.dataset import Dataset_selector

def load_pruned_model(model_path, device):
    """Load the pruned ResNet model"""
    print(f"Loading model from: {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    print("Model loaded successfully")
    return model

def test_model(model, test_loader, device):
    """Test the model and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    total_time = 0
    num_samples = 0
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            start_time = time.time()
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).squeeze()
            predictions = (probabilities > 0.5).float()
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            batch_time = time.time() - start_time
            total_time += batch_time
            num_samples += images.size(0)
            
            if batch_idx % 100 == 0:
                avg_time_per_sample = total_time / num_samples
                print(f"Batch {batch_idx}/{len(test_loader)}, "
                      f"Avg time per sample: {avg_time_per_sample:.4f}s")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    cm = confusion_matrix(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probabilities)
    
    avg_inference_time = total_time / num_samples
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'avg_inference_time': avg_inference_time,
        'total_samples': num_samples,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probabilities)
    }

def print_results(results, dataset_name):
    """Print formatted test results"""
    print(f"\n{'='*60}")
    print(f"TEST RESULTS FOR {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    print(f"Total Samples: {results['total_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"AUC-ROC: {results['auc']:.4f}")
    print(f"Average Inference Time: {results['avg_inference_time']:.4f} seconds/sample")
    print(f"FPS: {1.0/results['avg_inference_time']:.2f} samples/second")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              0     1")
    print(f"Actual   0  {results['confusion_matrix'][0,0]:4d} {results['confusion_matrix'][0,1]:4d}")
    print(f"         1  {results['confusion_matrix'][1,0]:4d} {results['confusion_matrix'][1,1]:4d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = results['confusion_matrix'].ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nAdditional Metrics:")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Specificity: {specificity:.4f}")

def save_results(results, dataset_name, output_dir):
    """Save results to files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save summary metrics
    summary = {
        'dataset': dataset_name,
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'auc': results['auc'],
        'avg_inference_time': results['avg_inference_time'],
        'total_samples': results['total_samples']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, f'{dataset_name}_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed predictions
    detailed_results = pd.DataFrame({
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'probability': results['probabilities']
    })
    
    detailed_path = os.path.join(output_dir, f'{dataset_name}_predictions.csv')
    detailed_results.to_csv(detailed_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"Summary: {summary_path}")
    print(f"Detailed predictions: {detailed_path}")

def main():
    parser = argparse.ArgumentParser(description='Test Pruned ResNet Model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pruned model (.pt file)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['hardfake', 'rvf10k', '140k', '190k', '200k', '330k'],
                        help='Dataset to test on')
    
    # Dataset-specific paths
    parser.add_argument('--hardfake_csv', type=str, default=None,
                        help='Path to hardfake CSV file')
    parser.add_argument('--hardfake_root', type=str, default=None,
                        help='Root directory for hardfake dataset')
    
    parser.add_argument('--rvf10k_train_csv', type=str, default=None,
                        help='Path to rvf10k train CSV file')
    parser.add_argument('--rvf10k_valid_csv', type=str, default=None,
                        help='Path to rvf10k validation CSV file')
    parser.add_argument('--rvf10k_root', type=str, default=None,
                        help='Root directory for rvf10k dataset')
    
    parser.add_argument('--realfake140k_train_csv', type=str, default=None,
                        help='Path to 140k train CSV file')
    parser.add_argument('--realfake140k_valid_csv', type=str, default=None,
                        help='Path to 140k validation CSV file')
    parser.add_argument('--realfake140k_test_csv', type=str, default=None,
                        help='Path to 140k test CSV file')
    parser.add_argument('--realfake140k_root', type=str, default=None,
                        help='Root directory for 140k dataset')
    
    parser.add_argument('--realfake190k_root', type=str, default=None,
                        help='Root directory for 190k dataset')
    
    parser.add_argument('--realfake200k_train_csv', type=str, default=None,
                        help='Path to 200k train CSV file')
    parser.add_argument('--realfake200k_val_csv', type=str, default=None,
                        help='Path to 200k validation CSV file')
    parser.add_argument('--realfake200k_test_csv', type=str, default=None,
                        help='Path to 200k test CSV file')
    parser.add_argument('--realfake200k_root', type=str, default=None,
                        help='Root directory for 200k dataset')
    
    parser.add_argument('--realfake330k_root', type=str, default=None,
                        help='Root directory for 330k dataset')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save results (default: ./test_results)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load model
    try:
        model = load_pruned_model(args.model_path, device,weights_only=False)
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create dataset
    try:
        dataset = Dataset_selector(
            dataset_mode=args.dataset,
            hardfake_csv_file=args.hardfake_csv,
            hardfake_root_dir=args.hardfake_root,
            rvf10k_train_csv=args.rvf10k_train_csv,
            rvf10k_valid_csv=args.rvf10k_valid_csv,
            rvf10k_root_dir=args.rvf10k_root,
            realfake140k_train_csv=args.realfake140k_train_csv,
            realfake140k_valid_csv=args.realfake140k_valid_csv,
            realfake140k_test_csv=args.realfake140k_test_csv,
            realfake140k_root_dir=args.realfake140k_root,
            realfake200k_train_csv=args.realfake200k_train_csv,
            realfake200k_val_csv=args.realfake200k_val_csv,
            realfake200k_test_csv=args.realfake200k_test_csv,
            realfake200k_root_dir=args.realfake200k_root,
            realfake190k_root_dir=args.realfake190k_root,
            realfake330k_root_dir=args.realfake330k_root,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            ddp=False
        )
        
        test_loader = dataset.loader_test
        print(f"Test dataset loaded successfully with {len(test_loader)} batches")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # Test model
    try:
        results = test_model(model, test_loader, device)
        
        # Print results
        print_results(results, args.dataset)
        
        # Save results
        save_results(results, args.dataset, args.output_dir)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return
    
    print(f"\nTesting completed successfully!")

if __name__ == '__main__':
    main()
