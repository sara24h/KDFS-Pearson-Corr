import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np

class FuzzyGatingNetwork(nn.Module):

    def __init__(self, num_models=5, input_size=256):
        super(FuzzyGatingNetwork, self).__init__()
        
        # Feature extractor سبک
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fuzzy Gating Head
        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_models)
        )
        
    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.gate(feat)
        fuzzy_weights = F.softmax(logits, dim=1)
        return fuzzy_weights


class MultiModelNormalization(nn.Module):
    """Normalization برای هر مدل با mean/std مخصوص"""
    def __init__(self, means, stds):
        super(MultiModelNormalization, self).__init__()
        self.num_models = len(means)
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            self.register_buffer(
                f'mean_{i}', 
                torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
            )
            self.register_buffer(
                f'std_{i}', 
                torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
            )
    
    def forward(self, x, model_idx):
        mean = getattr(self, f'mean_{model_idx}')
        std = getattr(self, f'std_{model_idx}')
        return (x - mean) / std


class FuzzyEnsembleModel(nn.Module):
    """ترکیب 5 مدل با Fuzzy Gating - مدل‌ها freeze"""
    def __init__(self, models, means, stds, freeze_models=True):
        super(FuzzyEnsembleModel, self).__init__()
        
        self.num_models = len(models)
        
        # مدل‌های اصلی (FREEZE)
        self.models = nn.ModuleList(models)
        if freeze_models:
            for model in self.models:
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
        
        self.normalizations = MultiModelNormalization(means, stds)
        self.gating_network = FuzzyGatingNetwork(num_models=self.num_models)
        
    def forward(self, x, return_individual=False):
        fuzzy_weights = self.gating_network(x)
    
        model_outputs = []
        for i, model in enumerate(self.models):
            x_normalized = self.normalizations(x, i)
            with torch.no_grad():
                output = model(x_normalized)
            
            # --- این قسمت حیاتیه ---
                if isinstance(output, (tuple, list)):
                    output = output[0]  # فقط logits (اولین خروجی)
            # ------------------------
            
            model_outputs.append(output)
    
        model_outputs = torch.stack(model_outputs, dim=1)
        weights_expanded = fuzzy_weights.unsqueeze(-1)
        final_output = (model_outputs * weights_expanded).sum(dim=1)
    
        if return_individual:
            return final_output, fuzzy_weights, model_outputs
        return final_output, fuzzy_weights


# =============================================================================
# لود مدل‌ها و DataLoaders
# =============================================================================

def load_pruned_models(model_paths, device):
    """لود 5 مدل هرس‌شده"""
    from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    
    models = []
    for i, path in enumerate(model_paths):
        print(f"Loading model {i+1}/5: {os.path.basename(path)}")
        
        checkpoint = torch.load(path, map_location=device)
        masks = checkpoint['masks']
        
        model = ResNet_50_pruned_hardfakevsreal(masks=masks)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   → Params: {total_params:,}")
        
        models.append(model)
    
    print(f"\nAll {len(models)} models loaded successfully!\n")
    return models


def create_data_loaders_kaggle(
    base_dir='/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k',
    batch_size=32,
    num_workers=2
):
   
    print("="*70)
    print("📂 آماده‌سازی DataLoaders...")
    print("="*70)
    print(f"Base directory: {base_dir}\n")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # چک کردن مسیرها
    train_path = os.path.join(base_dir, 'train')
    valid_path = os.path.join(base_dir, 'valid')
    test_path = os.path.join(base_dir, 'test')
    
    for path, name in [(train_path, 'train'), (valid_path, 'valid'), (test_path, 'test')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ پوشه {name} پیدا نشد: {path}")
        print(f"✅ {name}: {path}")
    
    print()
    
    # ساخت Datasets
    try:
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(valid_path, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(test_path, transform=val_test_transform)
    except Exception as e:
        print(f"❌ خطا: {e}")
        print("\n💡 ساختار باید دقیقاً اینجوری باشه:")
        print("   train/fake/  و  train/real/")
        print("   valid/fake/  و  valid/real/")
        print("   test/fake/   و  test/real/")
        raise
    
    # آمار
    print("📊 آمار دیتاست:")
    print(f"   Train:      {len(train_dataset):,} samples")
    print(f"   Validation: {len(val_dataset):,} samples")
    print(f"   Test:       {len(test_dataset):,} samples")
    print(f"   Total:      {len(train_dataset) + len(val_dataset) + len(test_dataset):,} samples")
    print(f"\n   Classes: {train_dataset.classes}")
    print(f"   Class mapping: {train_dataset.class_to_idx}\n")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ DataLoaders created!")
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader

def train_fuzzy_gating(ensemble_model, train_loader, val_loader, 
                       num_epochs=10, lr=1e-3, device='cuda', 
                       save_dir='/kaggle/working/checkpoints'):
    """آموزش فقط Gating Network"""
    os.makedirs(save_dir, exist_ok=True)
    
    trainable_params = list(ensemble_model.gating_network.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("="*70)
    print("🚀 شروع آموزش Fuzzy Gating Network")
    print("="*70)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}\n")
    
    for epoch in range(num_epochs):
        # Training
        ensemble_model.gating_network.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, fuzzy_weights = ensemble_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
        # Validation
        ensemble_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]  '):
                images, labels = images.to(device), labels.to(device)
                outputs, _ = ensemble_model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'gating_state_dict': ensemble_model.gating_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }
            save_path = os.path.join(save_dir, 'best_fuzzy_gating.pt')
            torch.save(checkpoint, save_path)
            print(f"   ✅ Best model saved! (Val Acc: {best_val_acc:.2f}%)")
        
        print("-"*70 + "\n")
    
    print("="*70)
    print(f"🎉 آموزش تمام شد! Best Val Acc: {best_val_acc:.2f}%")
    print("="*70)
    
    return best_val_acc, history


def evaluate_ensemble(ensemble_model, test_loader, device='cuda', dataset_name='Test'):
    """ارزیابی کامل و تحلیل وزن‌های فازی"""
    ensemble_model.eval()
    
    all_predictions = []
    all_labels = []
    all_fuzzy_weights = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Evaluating {dataset_name}'):
            images = images.to(device)
            
            outputs, fuzzy_weights, individual_outputs = ensemble_model(
                images, return_individual=True
            )
            
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_fuzzy_weights.append(fuzzy_weights.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = 100. * (all_predictions == all_labels).sum() / len(all_labels)
    
    all_fuzzy_weights = np.concatenate(all_fuzzy_weights, axis=0)
    avg_weights = all_fuzzy_weights.mean(axis=0)
    
    print("\n" + "="*70)
    print(f"📈 {dataset_name} Results:")
    print("="*70)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nمیانگین وزن‌های فازی:")
    for i, w in enumerate(avg_weights):
        print(f"   Model {i+1}: {w:.4f} ({w*100:.2f}%)")
    print("="*70)
    
    return accuracy, avg_weights


# =============================================================================
# Main Script - آماده برای Kaggle
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}\n")
    
    # ===== تنظیمات =====
    model_paths = [
        '/kaggle/input/10k-pearson-pruned/pytorch/default/1/10k_pearson_pruned.pt',
        '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
        '/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
        '/kaggle/input/200k-pearson-pruned/pytorch/default/1/200k_kdfs_pruned.pt',
        '/kaggle/input/330k-base-pruned/pytorch/default/1/330k_base_pruned.pt'
    ]
    
    means = [
        (0.5212, 0.4260, 0.3811),  # 10k
        (0.5207, 0.4258, 0.3806),  # 140k
        (0.4868, 0.3972, 0.3624),  # 200k
        (0.4668, 0.3816, 0.3414),  # 190k
        (0.4923, 0.4042, 0.3624)   # 330k
    ]
    
    stds = [
        (0.2486, 0.2238, 0.2211),
        (0.2490, 0.2239, 0.2212),
        (0.2296, 0.2066, 0.2009),
        (0.2410, 0.2161, 0.2081),
        (0.2446, 0.2198, 0.2141)
    ]
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    DATA_DIR = '/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k'
    
    # ===== لود مدل‌ها =====
    print("="*70)
    print("📦 لود مدل‌های هرس‌شده...")
    print("="*70)
    models = load_pruned_models(model_paths, device)
    
    # ===== ساخت Ensemble =====
    print("="*70)
    print("🔧 ساخت Fuzzy Ensemble Model...")
    print("="*70)
    ensemble_model = FuzzyEnsembleModel(
        models=models,
        means=means,
        stds=stds,
        freeze_models=True
    ).to(device)
    
    trainable_params = sum(p.numel() for p in ensemble_model.gating_network.parameters())
    total_params = sum(p.numel() for p in ensemble_model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable (Gating only): {trainable_params:,}")
    print(f"Frozen: {total_params - trainable_params:,}\n")
    
    # ===== آماده‌سازی داده =====
    train_loader, val_loader, test_loader = create_data_loaders_kaggle(
        base_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=2
    )
    
    # ===== آموزش =====
    best_acc, history = train_fuzzy_gating(
        ensemble_model=ensemble_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        save_dir='/kaggle/working/checkpoints'
    )
    
    # ===== ارزیابی روی Validation =====
    print("\n" + "="*70)
    print("🔍 ارزیابی روی Validation Set...")
    print("="*70)
    
    best_checkpoint = torch.load('/kaggle/working/checkpoints/best_fuzzy_gating.pt')
    ensemble_model.gating_network.load_state_dict(best_checkpoint['gating_state_dict'])
    
    val_acc, val_weights = evaluate_ensemble(ensemble_model, val_loader, device, 'Validation')
    
    # ===== ارزیابی نهایی روی Test =====
    print("\n" + "="*70)
    print("🏆 ارزیابی نهایی روی Test Set...")
    print("="*70)
    
    test_acc, test_weights = evaluate_ensemble(ensemble_model, test_loader, device, 'Test')
    
    # ===== خلاصه نهایی =====
    print("\n" + "="*70)
    print("📊 نتایج نهایی")
    print("="*70)
    print(f"✅ Best Validation Accuracy: {best_acc:.2f}%")
    print(f"✅ Final Test Accuracy: {test_acc:.2f}%")
    print("="*70)
    
    # ذخیره مدل نهایی
    final_save_path = '/kaggle/working/fuzzy_ensemble_final.pt'
    torch.save({
        'gating_state_dict': ensemble_model.gating_network.state_dict(),
        'val_acc': best_acc,
        'test_acc': test_acc,
        'val_weights': val_weights,
        'test_weights': test_weights,
        'history': history
    }, final_save_path)
    print(f"\n💾 مدل نهایی ذخیره شد: {final_save_path}")
