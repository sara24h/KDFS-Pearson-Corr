"""
FUZZY GATING ENSEMBLE - با دیتاست از قبل Split شده
=========================================================
این نسخه برای دیتاست‌هایی که از قبل به train/valid/test تقسیم شدن
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
from pathlib import Path
import glob

# ==================== CONFIGURATION ====================
class Config:

    MODEL_PATHS = [
        '/kaggle/input/10k-pearson-pruned/pytorch/default/1/10k_pearson_pruned.pt',
        '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
        '/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
        '/kaggle/input/200k-pearson-pruned/pytorch/default/1/200k_kdfs_pruned.pt',
        '/kaggle/input/330k-base-pruned/pytorch/default/1/330k_base_pruned.pt',
    ]
    
    MASKS_PATHS = None
    
    DATA_DIR = '/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k'
    
    # مسیر خروجی
    OUTPUT_DIR = 'kaggle/working/fuzzy_ensemble_output'
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Image size
    IMAGE_SIZE = 224
    
    # Random seed
    RANDOM_SEED = 42


# ==================== DATA LOADING ====================
def load_presplit_dataset(base_dir, image_size=224):

    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Helper function to load paths and labels
    def load_split(split_dir):
        paths = []
        labels = []
        
        # Real images (label = 0)
        real_dir = Path(split_dir) / 'real'
        if real_dir.exists():
            real_images = glob.glob(str(real_dir / '*.jpg')) + \
                          glob.glob(str(real_dir / '*.png')) + \
                          glob.glob(str(real_dir / '*.jpeg')) + \
                          glob.glob(str(real_dir / '*.JPG')) + \
                          glob.glob(str(real_dir / '*.PNG'))
            paths.extend(real_images)
            labels.extend([0] * len(real_images))
        else:
            print(f"  ⚠ Warning: {real_dir} not found")
            real_images = []
        
        # Fake images (label = 1)
        fake_dir = Path(split_dir) / 'fake'
        if fake_dir.exists():
            fake_images = glob.glob(str(fake_dir / '*.jpg')) + \
                          glob.glob(str(fake_dir / '*.png')) + \
                          glob.glob(str(fake_dir / '*.jpeg')) + \
                          glob.glob(str(fake_dir / '*.JPG')) + \
                          glob.glob(str(fake_dir / '*.PNG'))
            paths.extend(fake_images)
            labels.extend([1] * len(fake_images))
        else:
            print(f"  ⚠ Warning: {fake_dir} not found")
            fake_images = []
        
        split_name = Path(split_dir).name
        print(f"  {split_name:>5}: {len(real_images):>5} real + {len(fake_images):>5} fake = {len(paths):>6} total")
        
        return paths, labels
    
    print("\n📂 Loading pre-split dataset from:")
    print(f"   {base_dir}")
    print()
    
    base_path = Path(base_dir)
    
    # Load each split
    train_paths, train_labels = load_split(base_path / 'train')
    val_paths, val_labels = load_split(base_path / 'valid')
    test_paths, test_labels = load_split(base_path / 'test')
    
    if len(train_paths) == 0:
        raise ValueError("❌ No training data found! Check your DATA_DIR path.")
    if len(val_paths) == 0:
        raise ValueError("❌ No validation data found! Check your DATA_DIR path.")
    
    return (train_paths, train_labels, train_transform), \
           (val_paths, val_labels, val_transform), \
           (test_paths, test_labels, val_transform)


# ==================== MAIN PIPELINE ====================
def main():
    """پایپلاین اصلی"""
    
    print("\n" + "="*70)
    print("FUZZY GATING ENSEMBLE - COMPLETE PIPELINE")
    print("="*70)
    
    config = Config()
    
    # تنظیم seed برای reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    import numpy as np
    np.random.seed(config.RANDOM_SEED)
    
    # ==================== STEP 1: DATA PREPARATION ====================
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    # Load pre-split dataset
    train_data, val_data, test_data = load_presplit_dataset(
        base_dir=config.DATA_DIR,
        image_size=config.IMAGE_SIZE
    )
    
    train_paths, train_labels, train_transform = train_data
    val_paths, val_labels, val_transform = val_data
    test_paths, test_labels, test_transform = test_data
    
    # Load ensemble configuration for masks
    print("\n📦 Loading model masks and configurations...")
    from data_preparation import setup_ensemble_training, CustomDataset
    
    ensemble_config, masks_list, _, _ = setup_ensemble_training(
        model_paths=config.MODEL_PATHS,
        masks_paths=config.MASKS_PATHS,
        data_dir=None,  # We're not using this for data
        csv_path=None,
        output_dir=config.OUTPUT_DIR
    )
    
    # Create datasets
    train_dataset = CustomDataset(train_paths, train_labels, train_transform)
    val_dataset = CustomDataset(val_paths, val_labels, val_transform)
    test_dataset = CustomDataset(test_paths, test_labels, test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n✓ Data preparation complete!")
    print(f"  Train: {len(train_dataset):>6} images ({len(train_loader):>4} batches)")
    print(f"  Val:   {len(val_dataset):>6} images ({len(val_loader):>4} batches)")
    print(f"  Test:  {len(test_dataset):>6} images ({len(test_loader):>4} batches)")
    
    # Class distribution check
    from collections import Counter
    print(f"\n  Train distribution: Real={Counter(train_labels)[0]}, Fake={Counter(train_labels)[1]}")
    print(f"  Val distribution:   Real={Counter(val_labels)[0]}, Fake={Counter(val_labels)[1]}")
    print(f"  Test distribution:  Real={Counter(test_labels)[0]}, Fake={Counter(test_labels)[1]}")
    
    # ==================== STEP 2: CREATE ENSEMBLE MODEL ====================
    print("\n" + "="*70)
    print("STEP 2: CREATE ENSEMBLE MODEL")
    print("="*70)
    
    from fuzzy_ensemble import PrunedResNetEnsemble
    
    ensemble_model = PrunedResNetEnsemble(
        model_paths=config.MODEL_PATHS,
        masks_list=masks_list,
        means_stds=ensemble_config['means_stds'],
        num_features=2048
    )
    
    total_params = sum(p.numel() for p in ensemble_model.parameters())
    trainable_params = sum(p.numel() for p in ensemble_model.parameters() if p.requires_grad)
    
    print(f"\n✓ Ensemble model created!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    # ==================== STEP 3: TRAINING ====================
    print("\n" + "="*70)
    print("STEP 3: TRAINING FUZZY GATING")
    print("="*70)
    
    from fuzzy_ensemble import train_fuzzy_ensemble
    
    history = train_fuzzy_ensemble(
        ensemble_model=ensemble_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        device=config.DEVICE
    )
    
    print(f"\n✓ Training complete!")
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['val_acc'], label='Val Accuracy', color='green', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training History - Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(f'{config.OUTPUT_DIR}/training_history.png', dpi=300, bbox_inches='tight')
    print(f"  Training history saved to {config.OUTPUT_DIR}/training_history.png")
    plt.close()
    
    # ==================== STEP 4: LOAD BEST MODEL ====================
    print("\n" + "="*70)
    print("STEP 4: LOAD BEST MODEL")
    print("="*70)
    
    checkpoint = torch.load('best_fuzzy_ensemble.pt', map_location=config.DEVICE)
    ensemble_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Best model loaded (Val Acc: {checkpoint['val_acc']:.4f})")
    
    # ==================== STEP 5: ANALYZE GATING WEIGHTS ====================
    print("\n" + "="*70)
    print("STEP 5: ANALYZE GATING WEIGHTS")
    print("="*70)
    
    from fuzzy_ensemble import analyze_gating_weights
    
    weights, memberships = analyze_gating_weights(
        ensemble_model,
        val_loader,
        config.DEVICE,
        num_samples=1000
    )
    
    # ==================== STEP 6: LOAD SINGLE MODELS ====================
    print("\n" + "="*70)
    print("STEP 6: LOAD SINGLE MODELS FOR COMPARISON")
    print("="*70)
    
    from pruned_resnet import ResNet_50_pruned_hardfakevsreal
    
    single_models = []
    for i, (model_path, masks) in enumerate(zip(config.MODEL_PATHS, masks_list)):
        print(f"Loading Model {i+1}...")
        model = ResNet_50_pruned_hardfakevsreal(masks)
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        single_models.append(model)
    
    print(f"✓ Loaded {len(single_models)} single models")
    
    # ==================== STEP 7: COMPREHENSIVE EVALUATION ====================
    print("\n" + "="*70)
    print("STEP 7: COMPREHENSIVE EVALUATION ON TEST SET")
    print("="*70)
    
    from evaluation_comparison import run_complete_evaluation
    
    results, comparison_df, improvement = run_complete_evaluation(
        ensemble_model=ensemble_model,
        single_models=single_models,
        means_stds=ensemble_config['means_stds'],
        test_loader=test_loader,  # ⭐ استفاده از test set
        device=config.DEVICE,
        save_dir=f'{config.OUTPUT_DIR}/evaluation'
    )
    
    # ==================== STEP 8: FINAL SUMMARY ====================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    best_single_idx = comparison_df[comparison_df['Model'] != 'Ensemble']['Accuracy'].idxmax()
    best_single_acc = comparison_df[comparison_df['Model'] != 'Ensemble']['Accuracy'].max()
    ensemble_acc = comparison_df[comparison_df['Model'] == 'Ensemble']['Accuracy'].values[0]
    
    print(f"\n📊 Results on Test Set:")
    print(f"  Best Single Model: {comparison_df.loc[best_single_idx, 'Model']}")
    print(f"  Best Single Accuracy: {best_single_acc:.4f}")
    print(f"  Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\n✅ SUCCESS! Ensemble outperforms best single model by {improvement:.2f}%")
    else:
        print(f"\n⚠ Ensemble underperforms by {abs(improvement):.2f}%")
        print("  Consider:")
        print("  - Increasing training epochs")
        print("  - Adjusting learning rate")
        print("  - Modifying fuzzy membership functions")
        print("  - Adding more regularization")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {config.OUTPUT_DIR}")
    print("\nDataset used:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print("\nFiles generated:")
    print("  - config.json")
    print("  - masks_model_*.pt")
    print("  - best_fuzzy_ensemble.pt")
    print("  - training_history.png")
    print("  - evaluation/")
    print("    ├── comparison_table.csv")
    print("    ├── metrics_comparison.png")
    print("    ├── roc_curves.png")
    print("    ├── confusion_matrices.png")
    print("    ├── gating_weights.png")
    print("    └── detailed_results.pkl")
    print("\n" + "="*70 + "\n")
    
    return ensemble_model, results, comparison_df


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fuzzy Gating Ensemble Pipeline (Pre-split Dataset)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'infer'],
                       help='Operation mode')
    parser.add_argument('--model', type=str, default='best_fuzzy_ensemble.pt',
                       help='Path to model checkpoint (for eval/infer)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image (for infer mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training pipeline...")
        ensemble_model, results, comparison_df = main()
        
    elif args.mode == 'eval':
        print("Evaluation mode not fully implemented in this version.")
        print("Use 'train' mode which includes evaluation on test set.")
        
    elif args.mode == 'infer':
        if args.image is None:
            print("Error: --image required for inference mode")
            sys.exit(1)
        
        print("Inference mode not fully implemented in this version.")
        print("You can use the inference_on_new_image function from the original pipeline.")
    
    else:
        print("Usage:")
        print("  python main_pipeline_presplit.py --mode train")
