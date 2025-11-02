import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
import warnings
import argparse
import shutil
import json

warnings.filterwarnings("ignore")


# =============================================================================
# Fuzzy Hesitant Components
# =============================================================================

class HesitantFuzzyMembership(nn.Module):
    """
    Hesitant Fuzzy Set: هر مدل می‌تواند چندین مقدار عضویت داشته باشه
    که عدم قطعیت رو بهتر مدل می‌کنه
    """
    def __init__(self, input_dim: int, num_models: int, num_memberships: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_models = num_models
        self.num_memberships = num_memberships
        
        # شبکه برای تولید membership values
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # برای هر مدل، چندین membership value تولید می‌کنیم
        self.membership_generator = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_models * num_memberships)
        )
        
        # پارامترهای یادگیری برای aggregation
        self.aggregation_weights = nn.Parameter(torch.ones(num_memberships) / num_memberships)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            final_weights: وزن‌های نهایی برای هر مدل [batch, num_models]
            all_memberships: تمام membership values [batch, num_models, num_memberships]
        """
        features = self.feature_net(x).flatten(1)
        
        # تولید membership values
        memberships = self.membership_generator(features)
        memberships = memberships.view(-1, self.num_models, self.num_memberships)
        
        # نرمال‌سازی membership values (هر مقدار بین 0 و 1)
        memberships = torch.sigmoid(memberships)
        
        # Hesitant Fuzzy Aggregation با وزن‌های یادگیری‌شده
        agg_weights = F.softmax(self.aggregation_weights, dim=0)
        final_weights = (memberships * agg_weights.view(1, 1, -1)).sum(dim=2)
        
        # نرمال‌سازی نهایی
        final_weights = F.softmax(final_weights, dim=1)
        
        return final_weights, memberships


class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class FuzzyHesitantEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]], 
                 stds: List[Tuple[float]], num_memberships: int = 3, freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128, 
            num_models=self.num_models,
            num_memberships=num_memberships
        )
        
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor, return_details: bool = False):
        # محاسبه Hesitant Fuzzy Weights
        final_weights, all_memberships = self.hesitant_fuzzy(x)
        
        # پیش‌بینی مدل‌ها
        outputs = []
        for i, model in enumerate(self.models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = model(x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)  # [batch, num_models, 1]
        
        # Ensemble با Hesitant Fuzzy Weights
        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
        
        if return_details:
            return final_output, final_weights, all_memberships, outputs
        return final_output, final_weights


# =============================================================================
# لود مدل‌ها
# =============================================================================

def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    try:
        from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal. Ensure model.pruned_model.ResNet_pruned is available.")

    models = []
    print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f"  [WARNING] File not found: {path}")
            continue
        print(f"  [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            param_count = sum(p.numel() for p in model.parameters())
            print(f"     → Parameters: {param_count:,}")
            models.append(model)
        except Exception as e:
            print(f"  [ERROR] Failed to load {path}: {e}")
            continue
    if len(models) == 0:
        raise ValueError("No models loaded!")
    print(f"All {len(models)} models loaded!\n")
    return models


# =============================================================================
# دیتالودرها
# =============================================================================

def create_dataloaders(base_dir: str, batch_size: int, num_workers: int = 2):
    print("="*70)
    print("Creating DataLoaders...")
    print("="*70)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    splits = ['train', 'valid', 'test']
    datasets_dict = {}
    for split in splits:
        path = os.path.join(base_dir, split)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")
        print(f"{split.capitalize():5}: {path}")
        transform = train_transform if split == 'train' else val_test_transform
        datasets_dict[split] = datasets.ImageFolder(path, transform=transform)

    print(f"\nDataset Stats:")
    for split, ds in datasets_dict.items():
        print(f"  {split.capitalize():5}: {len(ds):,} images | Classes: {ds.classes}")
    print(f"  Class → Index: {datasets_dict['train'].class_to_idx}\n")

    loaders = {}
    for split, ds in datasets_dict.items():
        shuffle = (split == 'train')
        drop_last = (split != 'test')
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=drop_last
        )

    print(f"DataLoaders ready! Batch size: {batch_size}")
    print(f"  Batches → Train: {len(loaders['train'])}, Val: {len(loaders['valid'])}, Test: {len(loaders['test'])}")
    print("="*70 + "\n")
    return loaders['train'], loaders['valid'], loaders['test']


# =============================================================================
# ارزیابی مدل تکی
# =============================================================================

@torch.no_grad()
def evaluate_single_model(model: nn.Module, loader: DataLoader, device: torch.device, name: str) -> float:
    model.eval()
    correct = total = 0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
        images, labels = images.to(device), labels.to(device).float()
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
    acc = 100. * correct / total
    print(f"   {name}: {acc:.2f}%")
    return acc


# =============================================================================
# آموزش Hesitant Fuzzy Network
# =============================================================================

def train_hesitant_fuzzy(ensemble_model, train_loader, val_loader, num_epochs, lr, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    hesitant_net = ensemble_model.module.hesitant_fuzzy if isinstance(ensemble_model, nn.DataParallel) else ensemble_model.hesitant_fuzzy

    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'membership_variance': []}

    print("="*70)
    print("Training Fuzzy Hesitant Network (DataParallel)")
    print("="*70)
    print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
    print(f"Epochs: {num_epochs} | Initial LR: {lr} | Device: {device}")
    print(f"Hesitant memberships per model: {hesitant_net.num_memberships}\n")

    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loss = train_correct = train_total = 0.0
        membership_vars = []
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs, weights, memberships, _ = ensemble_model(images, return_details=True)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += batch_size
            
            # محاسبه variance در membership values (نشان‌دهنده hesitancy)
            membership_vars.append(memberships.var(dim=2).mean().item())

            current_acc = 100. * train_correct / train_total
            avg_loss = train_loss / train_total
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{current_acc:.2f}%'})

        train_acc = 100. * train_correct / train_total
        train_loss /= train_total
        avg_membership_var = np.mean(membership_vars)
        val_acc = evaluate_accuracy(ensemble_model, val_loader, device)

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['membership_variance'].append(avg_membership_var)

        print(f"\nEpoch {epoch+1}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   Membership Variance (Hesitancy): {avg_membership_var:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            tmp_path = os.path.join(save_dir, 'best_tmp.pt')
            final_path = os.path.join(save_dir, 'best_hesitant_fuzzy.pt')
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'hesitant_state_dict': hesitant_net.state_dict(),
                    'val_acc': val_acc,
                    'history': history
                }, tmp_path)
                if os.path.exists(tmp_path):
                    shutil.move(tmp_path, final_path)
                    print(f"   Best model saved → {val_acc:.2f}%")
                else:
                    print(f"   [WARNING] Failed to save model: {tmp_path} not found")
            except Exception as e:
                print(f"   [ERROR] Failed to save model: {e}")

        print("-" * 70)

    print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, history


# =============================================================================
# ارزیابی ensemble + membership analysis
# =============================================================================

@torch.no_grad()
def evaluate_ensemble_final(model, loader, device, name, model_names):
    model.eval()
    all_preds, all_labels, all_weights, all_memberships = [], [], [], []

    for images, labels in tqdm(loader, desc=f'Evaluating {name}'):
        images, labels = images.to(device), labels.to(device)
        outputs, weights, memberships, _ = model(images, return_details=True)
        pred = (outputs.squeeze(1) > 0).long()

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_weights.append(weights.cpu().numpy())
        all_memberships.append(memberships.cpu().numpy())

    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    avg_weights = np.concatenate(all_weights).mean(axis=0)
    avg_memberships = np.concatenate(all_memberships).mean(axis=0)

    print(f"\n{name} Results → Accuracy: {acc:.2f}%")
    print("\nFinal Hesitant Fuzzy Weights:")
    for i, (w, name) in enumerate(zip(avg_weights, model_names)):
        print(f"   Model {i+1} ({name}): {w:.4f} ({w*100:.2f}%)")
    
    print("\nHesitant Membership Values per Model:")
    for i, name in enumerate(model_names):
        memberships = avg_memberships[i]
        variance = memberships.var()
        print(f"   Model {i+1} ({name}):")
        print(f"      Memberships: {[f'{m:.3f}' for m in memberships]}")
        print(f"      Variance (Hesitancy): {variance:.4f}")
    
    return acc, avg_weights.tolist(), avg_memberships.tolist()


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
    return 100. * correct / total


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Fuzzy Hesitant Ensemble")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_memberships', type=int, default=3, help='Number of membership values per model')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    print(f"Device: {device} | GPUs: {gpu_count} | Using DataParallel")

    # مدل‌ها
    MODEL_PATHS = [
        '/kaggle/input/10k-pearson-pruned/pytorch/default/1/10k_pearson_pruned.pt',
        '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
        '/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
        '/kaggle/input/200k-pearson-pruned/pytorch/default/1/200k_kdfs_pruned.pt',
        '/kaggle/input/330k-pearson-pruned/pytorch/default/1/330k_pearson_pruned.pt'
    ]
    MODEL_NAMES = ["10k_pearson", "140k_pearson", "190k_pearson", "200k_kdfs", "330k_pearson"]
    MEANS = [(0.5212,0.4260,0.3811), (0.5207,0.4258,0.3806), (0.4868,0.3972,0.3624),
             (0.4668,0.3816,0.3414), (0.4923,0.4042,0.3624)]
    STDS = [(0.2486,0.2238,0.2211), (0.2490,0.2239,0.2212), (0.2296,0.2066,0.2009),
            (0.2410,0.2161,0.2081), (0.2446,0.2198,0.2141)]

    # لود مدل‌ها
    base_models = load_pruned_models(MODEL_PATHS, device)
    if len(base_models) != len(MODEL_PATHS):
        print(f"[WARNING] Only {len(base_models)}/{len(MODEL_PATHS)} models loaded. Adjusting MEANS/STDS.")
        MEANS = MEANS[:len(base_models)]
        STDS = STDS[:len(base_models)]
        MODEL_NAMES = MODEL_NAMES[:len(base_models)]

    # ساخت Hesitant Ensemble
    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS, 
        num_memberships=args.num_memberships,
        freeze_models=True
    ).to(device)
    
    if gpu_count > 1:
        ensemble = nn.DataParallel(ensemble)

    # آمار پارامترها
    hesitant_net = ensemble.module.hesitant_fuzzy if isinstance(ensemble, nn.DataParallel) else ensemble.hesitant_fuzzy
    trainable = sum(p.numel() for p in hesitant_net.parameters())
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"Total params: {total_params:,} | Trainable: {trainable:,} | Frozen: {total_params - trainable:,}\n")

    # دیتالودر
    train_loader, val_loader, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    # ارزیابی تکی قبل از آموزش
    print("\n" + "="*70)
    print("EVALUATING INDIVIDUAL MODELS ON TEST SET (Before Training)")
    print("="*70)
    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model(model, test_loader, device, f"Model {i+1} ({MODEL_NAMES[i]})")
        individual_accs.append(acc)
    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)
    print(f"\nBest Single Model: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")

    # آموزش Hesitant Fuzzy
    best_val_acc, history = train_hesitant_fuzzy(
        ensemble, train_loader, val_loader, 
        args.epochs, args.lr, device, args.save_dir
    )

    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ensemble, nn.DataParallel):
            ensemble.module.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        else:
            ensemble.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        print("Best hesitant fuzzy network loaded.\n")

    # ارزیابی ensemble
    print("\n" + "="*70)
    print("EVALUATING FUZZY HESITANT ENSEMBLE")
    print("="*70)
    ensemble_test_acc, ensemble_weights, membership_values = evaluate_ensemble_final(
        ensemble, test_loader, device, "Test", MODEL_NAMES
    )

    # مقایسه
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Best Single Model Acc : {best_single:.2f}%")
    print(f"Hesitant Ensemble Acc : {ensemble_test_acc:.2f}%")
    improvement = ensemble_test_acc - best_single
    print(f"Improvement           : {improvement:+.2f}%")

    # ذخیره نتایج
    results = {
        "method": "Fuzzy Hesitant Sets",
        "num_memberships": args.num_memberships,
        "individual_accuracies": {MODEL_NAMES[i]: acc for i, acc in enumerate(individual_accs)},
        "best_single": {"name": MODEL_NAMES[best_idx], "acc": best_single},
        "ensemble": {
            "acc": ensemble_test_acc, 
            "weights": ensemble_weights,
            "membership_values": membership_values
        },
        "improvement": improvement
    }
    result_path = '/kaggle/working/hesitant_fuzzy_results.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    # ذخیره مدل نهایی
    final_model_path = '/kaggle/working/hesitant_fuzzy_final.pt'
    torch.save({
        'hesitant_state_dict': hesitant_net.state_dict(),
        'results': results
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")
    print("All done!")


if __name__ == "__main__":
    main()
