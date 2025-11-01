import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np


class FuzzyGatingNetwork(nn.Module):
    def __init__(self, num_models=5):
        super(FuzzyGatingNetwork, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.gate = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, num_models)
        )
        
    def forward(self, x):
        x = self.features(x).flatten(1)
        return F.softmax(self.gate(x), dim=1)


class MultiModelNormalization(nn.Module):
    def __init__(self, means, stds):
        super(MultiModelNormalization, self).__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))
    def forward(self, x, idx):
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class FuzzyEnsembleModel(nn.Module):
    def __init__(self, models, means, stds, freeze_models=True):
        super(FuzzyEnsembleModel, self).__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        if freeze_models:
            for m in self.models:
                m.eval()
                for p in m.parameters(): p.requires_grad = False
        self.normalizations = MultiModelNormalization(means, stds)
        self.gating_network = FuzzyGatingNetwork(num_models=self.num_models)
        
    def forward(self, x, return_individual=False):
        weights = self.gating_network(x)  # (B, N)
        outputs = []
        for i, model in enumerate(self.models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = model(x_n)
                if isinstance(out, (tuple, list)): out = out[0]
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)  # (B, N, 1)
        final = (outputs * weights.unsqueeze(-1)).sum(dim=1)
        return (final, weights, outputs) if return_individual else (final, weights)


# =============================================================================
# لود مدل‌ها و DataLoaders
# =============================================================================

def load_pruned_models(model_paths, device):
    from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    models = []
    for i, path in enumerate(model_paths):
        print(f"Loading model {i+1}/5: {os.path.basename(path)}")
        ckpt = torch.load(path, map_location=device)
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()
        print(f"   → Params: {sum(p.numel() for p in model.parameters()):,}")
        models.append(model)
    print(f"\nAll {len(models)} models loaded successfully!\n")
    return models


def create_data_loaders_kaggle(base_dir, batch_size, num_workers=4):
    print("="*70)
    print("آماده‌سازی DataLoaders...")
    print("="*70)
    print(f"Base directory: {base_dir}\n")
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
    ])
    val_test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    paths = {'train': 'train', 'valid': 'valid', 'test': 'test'}
    for name, sub in paths.items():
        path = os.path.join(base_dir, sub)
        if not os.path.exists(path):
            raise FileNotFoundError(f"پوشه {name} پیدا نشد: {path}")
        print(f"{name}: {path}")

    train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(base_dir, 'valid'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(base_dir, 'test'), transform=val_test_transform)

    print(f"\nآمار دیتاست:")
    print(f"   Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    print(f"   Classes: {train_dataset.classes} | Mapping: {train_dataset.class_to_idx}\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"DataLoaders created! Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
    print("="*70 + "\n")
    return train_loader, val_loader, test_loader


# =============================================================================
# آموزش و ارزیابی
# =============================================================================

def train_fuzzy_gating(ensemble_model, train_loader, val_loader, num_epochs=10, lr=1e-3,
                       device='cuda', save_dir='/kaggle/working/checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    
    # --- دسترسی ایمن به gating_network ---
    gating_net = ensemble_model.module.gating_network if hasattr(ensemble_model, 'module') else ensemble_model.gating_network
    optimizer = torch.optim.AdamW(gating_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("="*70)
    print("شروع آموزش Fuzzy Gating Network (BCE)")
    print("="*70)
    print(f"Trainable params: {sum(p.numel() for p in gating_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | LR: {lr} | Device: {device}\n")

    for epoch in range(num_epochs):
        # Train
        ensemble_model.train()
        train_loss = train_correct = train_total = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]'):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs, _ = ensemble_model(images)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pred = (outputs.squeeze(1) > 0).long()
            train_total += labels.size(0)
            train_correct += pred.eq(labels.long()).sum().item()

        train_acc = 100. * train_correct / train_total
        train_loss /= train_total

        # Val
        ensemble_model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]  '):
                images, labels = images.to(device), labels.to(device).float()
                outputs, _ = ensemble_model(images)
                pred = (outputs.squeeze(1) > 0).long()
                val_total += labels.size(0)
                val_correct += pred.eq(labels.long()).sum().item()
        val_acc = 100. * val_correct / val_total

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'gating_state_dict': gating_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, os.path.join(save_dir, 'best_fuzzy_gating.pt'))
            print(f"   Best model saved! ({best_val_acc:.2f}%)")
        print("-"*70 + "\n")

    print(f"آموزش تمام شد! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, history


def evaluate_ensemble(ensemble_model, loader, device, name):
    ensemble_model.eval()
    preds, labels_list, weights_list = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f'Evaluating {name}'):
            images, labels = images.to(device), labels.to(device)
            outputs, weights, _ = ensemble_model(images, return_individual=True)
            pred = (outputs.squeeze(1) > 0).long()
            preds.extend(pred.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            weights_list.append(weights.cpu().numpy())
    
    acc = 100. * np.mean(np.array(preds) == np.array(labels_list))
    avg_weights = np.concatenate(weights_list).mean(axis=0)
    
    print("\n" + "="*70)
    print(f"{name} Results: Accuracy = {acc:.2f}%")
    print("میانگین وزن‌های فازی:")
    for i, w in enumerate(avg_weights):
        print(f"   Model {i+1}: {w:.4f} ({w*100:.2f}%)")
    print("="*70)
    return acc, avg_weights


# =============================================================================
# Main - با پشتیبانی کامل از 2 GPU
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"Using device: {device} | GPUs: {gpu_count}\n")

    # تنظیمات
    model_paths = [
        '/kaggle/input/10k-pearson-pruned/pytorch/default/1/10k_pearson_pruned.pt',
        '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
        '/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
        '/kaggle/input/200k-pearson-pruned/pytorch/default/1/200k_kdfs_pruned.pt',
        '/kaggle/input/330k-base-pruned/pytorch/default/1/330k_base_pruned.pt'
    ]
    means = [(0.5212,0.4260,0.3811), (0.5207,0.4258,0.3806), (0.4868,0.3972,0.3624),
             (0.4668,0.3816,0.3414), (0.4923,0.4042,0.3624)]
    stds = [(0.2486,0.2238,0.2211), (0.2490,0.2239,0.2212), (0.2296,0.2066,0.2009),
            (0.2410,0.2161,0.2081), (0.2446,0.2198,0.2141)]
    BATCH_SIZE = 32 * max(1, gpu_count)
    DATA_DIR = '/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k'

    # لود مدل‌ها
    models = load_pruned_models(model_paths, device)

    # ساخت مدل
    ensemble_model = FuzzyEnsembleModel(models, means, stds, freeze_models=True).to(device)
    if gpu_count > 1:
        print(f"Using {gpu_count} GPUs with DataParallel!")
        ensemble_model = nn.DataParallel(ensemble_model)

    # محاسبه پارامترها
    gating_net = ensemble_model.module.gating_network if hasattr(ensemble_model, 'module') else ensemble_model.gating_network
    trainable_params = sum(p.numel() for p in gating_net.parameters())
    total_params = sum(p.numel() for p in (ensemble_model.module.parameters() if hasattr(ensemble_model, 'module') else ensemble_model.parameters()))
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {total_params - trainable_params:,}\n")

    # دیتالودرها
    train_loader, val_loader, test_loader = create_data_loaders_kaggle(DATA_DIR, BATCH_SIZE, num_workers=4)

    # آموزش
    best_acc, history = train_fuzzy_gating(ensemble_model, train_loader, val_loader,
                                           num_epochs=10, lr=1e-3, device=device)

    # بارگذاری بهترین مدل
    best_ckpt = torch.load('/kaggle/working/checkpoints/best_fuzzy_gating.pt', map_location=device)
    gating_net.load_state_dict(best_ckpt['gating_state_dict'])

    # ارزیابی
    print("\n" + "="*70)
    print("ارزیابی روی Validation...")
    val_acc, val_weights = evaluate_ensemble(ensemble_model, val_loader, device, 'Validation')

    print("\n" + "="*70)
    print("ارزیابی نهایی روی Test...")
    test_acc, test_weights = evaluate_ensemble(ensemble_model, test_loader, device, 'Test')

    # ذخیره نهایی
    final_path = '/kaggle/working/fuzzy_ensemble_final.pt'
    torch.save({
        'gating_state_dict': gating_net.state_dict(),
        'val_acc': best_acc, 'test_acc': test_acc,
        'val_weights': val_weights, 'test_weights': test_weights,
        'history': history
    }, final_path)
    print(f"\nمدل نهایی ذخیره شد: {final_path}")
