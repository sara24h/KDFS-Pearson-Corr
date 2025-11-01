"""
FUZZY GATING ENSEMBLE - با دیتاست از قبل Split شده (اصلاح شده - بدون circular import)
=========================================================
این نسخه برای دیتاست‌هایی که از قبل به train/valid/test تقسیم شدن
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import sys
import os
from pathlib import Path
import glob
from collections import Counter
import numpy as np

# Import model classes from separate file
from fuzzy_ensemble_model import PrunedResNetEnsemble, train_fuzzy_ensemble, analyze_gating_weights

# ==================== CONFIGURATION ====================
class Config:
    MODEL_PATHS = [
        '/kaggle/input/10k-pearson-pruned/pytorch/default/1/10k_pearson_pruned.pt',
        '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
        '/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
        '/kaggle/input/200k-pearson-pruned/pytorch/default/1/200k_kdfs_pruned.pt',
        '/kaggle/input/330k-base-pruned/pytorch/default/1/330k_base_pruned.pt',
    ]
    
    MASKS_PATHS = None  # اگر فایل‌های mask دارید، مسیرها را اینجا قرار دهید
    
    DATA_DIR = '/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k'
    
    # مسیر خروجی
    OUTPUT_DIR = '/kaggle/working/fuzzy_ensemble_output'
    
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


# ==================== CUSTOM DATASET ====================
class CustomDataset(Dataset):
    """Dataset کاستوم برای بارگذاری تصاویر"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==================== DATA LOADING ====================
def load_presplit_dataset(base_dir, image_size=224):
    """بارگذاری دیتاست از قبل تقسیم شده"""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    def load_split(split_dir):
        paths = []
        labels = []
        
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
    
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # ==================== STEP 1: DATA PREPARATION ====================
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    train_data, val_data, test_data = load_presplit_dataset(
        base_dir=config.DATA_DIR,
        image_size=config.IMAGE_SIZE
    )
    
    train_paths, train_labels, train_transform = train_data
    val_paths, val_labels, val_transform = val_data
    test_paths, test_labels, test_transform = test_data
    
    print("\n📦 Loading model masks and configurations...")
    
    if config.MASKS_PATHS:
        print("  Loading masks from files...")
        masks_list = []
        for i, mask_path in enumerate(config.MASKS_PATHS):
            try:
                mask = torch.load(mask_path, map_location='cpu')
                masks_list.append(mask)
                print(f"    ✓ Mask {i+1} loaded from {mask_path}")
            except Exception as e:
                print(f"    ✗ Error loading mask {i+1}: {e}")
                masks_list.append(None)
    else:
        print("  No mask paths provided, using None for all models")
        masks_list = [None] * len(config.MODEL_PATHS)
 
    means_stds = [
        # Model 1: 10k_pearson_pruned
        ([0.5212, 0.4260, 0.3811], [0.2486, 0.2238, 0.2211]),  # 👈 مقادیر مدل 1 را اینجا بگذارید
        
        # Model 2: 140k_pearson_pruned
        ([0.5207, 0.4258, 0.3806], [0.2490, 0.2239, 0.2212]),  # 👈 مقادیر مدل 2 را اینجا بگذارید
        
        # Model 3: 190k_pearson_pruned
        ([0.4868, 0.3972, 0.3624], [0.2296, 0.2066, 0.2009]),  # 👈 مقادیر مدل 3 را اینجا بگذارید
        
        # Model 4: 200k_kdfs_pruned
        ([0.4668, 0.3816, 0.3414], [0.2410, 0.2161, 0.2081]),  # 👈 مقادیر مدل 4 را اینجا بگذارید
        
        # Model 5: 330k_base_pruned
        ([0.4923, 0.4042, 0.3624], [0.2446, 0.2198, 0.2141]),  # 👈 مقادیر مدل 5 را اینجا بگذارید
    ]
    
    ensemble_config = {
        'means_stds': means_stds
    }
    
    print("\n  📊 Mean/Std for each model:")
    for i, (mean, std) in enumerate(means_stds):
        print(f"    Model {i+1}: mean={mean}, std={std}")
    
    print(f"  ✓ Loaded configuration for {len(config.MODEL_PATHS)} models")
    
    train_dataset = CustomDataset(train_paths, train_labels, train_transform)
    val_dataset = CustomDataset(val_paths, val_labels, val_transform)
    test_dataset = CustomDataset(test_paths, test_labels, test_transform)
    
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
    
    print(f"\n  Train distribution: Real={Counter(train_labels)[0]}, Fake={Counter(train_labels)[1]}")
    print(f"  Val distribution:   Real={Counter(val_labels)[0]}, Fake={Counter(val_labels)[1]}")
    print(f"  Test distribution:  Real={Counter(test_labels)[0]}, Fake={Counter(test_labels)[1]}")
    
    # ==================== STEP 2: CREATE ENSEMBLE MODEL ====================
    print("\n" + "="*70)
    print("STEP 2: CREATE ENSEMBLE MODEL")
    print("="*70)
    
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
    
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
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
        test_loader=test_loader,
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
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {config.OUTPUT_DIR}")
    
    return ensemble_model, results, comparison_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fuzzy Gating Ensemble Pipeline')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'infer'],
                       help='Operation mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training pipeline...")
        ensemble_model, results, comparison_df = main()
    else:
        print(f"Mode '{args.mode}' not fully implemented.")
