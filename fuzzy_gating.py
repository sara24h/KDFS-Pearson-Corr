# fuzzy_gating_ddp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
import warnings
import argparse
warnings.filterwarnings("ignore")


# =============================================================================
# مدل‌های اصلی
# =============================================================================

class FuzzyGatingNetwork(nn.Module):
    def __init__(self, num_models: int = 5, dropout: float = 0.3):
        super().__init__()
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
            nn.Dropout(dropout), nn.Linear(64, num_models)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return F.softmax(self.gate(x), dim=1)


class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


class FuzzyEnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]], stds: List[Tuple[float]], freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.gating_network = FuzzyGatingNetwork(num_models=self.num_models)

        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor, return_individual: bool = False):
        weights = self.gating_network(x)
        outputs = []

        for i, model in enumerate(self.models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = model(x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        final_output = (outputs * weights.unsqueeze(-1)).sum(dim=1)

        if return_individual:
            return final_output, weights, outputs
        return final_output, weights


# =============================================================================
# تابع کمکی
# =============================================================================

def get_gating_network(model: nn.Module) -> nn.Module:
    if isinstance(model, DDP):
        return model.module.gating_network
    return model.gating_network


# =============================================================================
# لود مدل‌ها
# =============================================================================

def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

    models = []
    print(f"Rank {dist.get_rank() if dist.is_initialized() else 0} | Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if dist.get_rank() == 0:
            print(f"  [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        ckpt = torch.load(path, map_location='cpu')
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device).eval()

        if hasattr(model, 'masks'):
            for j, mask in enumerate(model.masks):
                mask = mask.to(device)

        param_count = sum(p.numel() for p in model.parameters())
        if dist.get_rank() == 0:
            print(f"     → Parameters: {param_count:,}")
        models.append(model)
    if dist.get_rank() == 0:
        print(f"All {len(models)} models loaded successfully!\n")
    return models


# =============================================================================
# دیتالودرها با DistributedSampler
# =============================================================================

def create_dataloaders(base_dir: str, batch_size: int, world_size: int, rank: int, num_workers: int = 4):
    print(f"Rank {rank} | Creating DataLoaders...")

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
        if rank == 0:
            print(f"  {split.capitalize():5}: {path}")
        transform = train_transform if split == 'train' else val_test_transform
        datasets_dict[split] = datasets.ImageFolder(path, transform=transform)

    if rank == 0:
        print(f"\nDataset Stats:")
        for split, ds in datasets_dict.items():
            print(f"  {split.capitalize():5}: {len(ds):,} images | Classes: {ds.classes}")
        print(f"  Class → Index: {datasets_dict['train'].class_to_idx}\n")

    loaders = {}
    for split, ds in datasets_dict.items():
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank) if world_size > 1 else None
        shuffle = (split == 'train') and (sampler is None)
        drop_last = (split == 'train')

        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, num_workers=num_workers,
            pin_memory=True, drop_last=drop_last
        )

    if rank == 0:
        print(f"DataLoaders ready! Per-GPU batch size: {batch_size} | World size: {world_size}")
        print(f"  Batches per GPU → Train: {len(loaders['train'])}, Val: {len(loaders['valid'])}, Test: {len(loaders['test'])}")
        print("="*70 + "\n")

    return loaders['train'], loaders['valid'], loaders['test']


# =============================================================================
# ارزیابی دقت
# =============================================================================

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
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
# آموزش Gating Network
# =============================================================================

def train_gating_network(
    ensemble_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    save_dir: str,
    rank: int
) -> Tuple[float, Dict[str, Any]]:

    os.makedirs(save_dir, exist_ok=True) if rank == 0 else None
    gating_net = get_gating_network(ensemble_model)
    optimizer = torch.optim.AdamW(gating_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    if rank == 0:
        print("="*70)
        print("Training Fuzzy Gating Network (DDP)")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in gating_net.parameters()):,}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr} | Device: {device}\n")

    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loader.sampler.set_epoch(epoch) if hasattr(train_loader, 'sampler') and train_loader.sampler else None

        train_loss = train_correct = train_total = 0.0
        progress_bar = tqdm(train_loader, desc=f'Rank {rank} | Epoch {epoch+1}/{num_epochs} [Train]') if rank == 0 else train_loader

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs, _ = ensemble_model(images)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += batch_size

            if rank == 0:
                current_acc = 100. * train_correct / train_total
                avg_loss = train_loss / train_total
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{current_acc:.2f}%'})

        # فقط rank 0 ارزیابی val
        if rank == 0:
            train_acc = 100. * train_correct / train_total
            train_loss /= train_total
            val_acc = evaluate_accuracy(ensemble_model, val_loader, device)

            scheduler.step()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"\nEpoch {epoch+1}:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = os.path.join(save_dir, 'best_fuzzy_gating.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'gating_state_dict': gating_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': history
                }, ckpt_path)
                print(f"   Best model saved → {val_acc:.2f}%")
            print("-" * 70)

    if rank == 0:
        print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, history if rank == 0 else (0, {})


# =============================================================================
# ارزیابی دقیق
# =============================================================================

@torch.no_grad()
def evaluate_ensemble_detailed(model: nn.Module, loader: DataLoader, device: torch.device, name: str, rank: int):
    if rank != 0:
        return 0, [], []

    model.eval()
    individual_preds = [[] for _ in range(model.module.num_models)]
    ensemble_preds = []
    all_labels = []
    all_weights = []

    print(f"\nEvaluating {name} with detailed per-model analysis...")
    for images, labels in tqdm(loader, desc=f'{name} Batch'):
        images, labels = images.to(device), labels.to(device)
        final_output, weights, individual_outputs = model(images, return_individual=True)
        ensemble_pred = (final_output.squeeze(1) > 0).long()
        ensemble_preds.extend(ensemble_pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_weights.append(weights.cpu().numpy())

        for i in range(model.module.num_models):
            pred_i = (individual_outputs[:, i].squeeze(1) > 0).long()
            individual_preds[i].extend(pred_i.cpu().numpy())

    all_labels = np.array(all_labels)
    ensemble_preds = np.array(ensemble_preds)
    individual_preds = [np.array(preds) for preds in individual_preds]
    avg_weights = np.concatenate(all_weights).mean(axis=0)
    ensemble_acc = 100. * np.mean(ensemble_preds == all_labels)

    print("\n" + "="*80)
    print(f"{name.upper()} RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'Accuracy':<10} {'Weight':<10} {'Status'}")
    print("-" * 80)
    individual_accs = []
    for i in range(model.module.num_models):
        acc_i = 100. * np.mean(individual_preds[i] == all_labels)
        individual_accs.append(acc_i)
        status = "STRONG" if avg_weights[i] > 0.2 else "WEAK" if avg_weights[i] < 0.1 else "MEDIUM"
        print(f"Model {i+1:<11} {acc_i:6.2f}%    {avg_weights[i]*100:6.2f}%    {status}")
    print("-" * 80)
    print(f"{'ENSEMBLE':<15} {ensemble_acc:6.2f}%    {'100.00%':<10}  FINAL")
    print("="*80)
    best_single = max(individual_accs)
    improvement = ensemble_acc - best_single
    print(f"Best single model: {best_single:.2f}%")
    print(f"Ensemble improvement: +{improvement:.2f}%")

    return ensemble_acc, individual_accs, avg_weights


# =============================================================================
# Main Worker
# =============================================================================

def main_worker(rank: int, world_size: int, args):
    # تنظیم DDP — اصلاح برای Kaggle
    if world_size > 1:
        # استفاده از 127.0.0.1 و پورت ثابت
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'  # پورت آزاد در Kaggle
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # تنظیمات
    MODEL_PATHS = [
        '/kaggle/input/10k-pearson-pruned/pytorch/default/1/10k_pearson_pruned.pt',
        '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
        '/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
        '/kaggle/input/200k-pearson-pruned/pytorch/default/1/200k_kdfs_pruned.pt',
        '/kaggle/input/330k-pearson-pruned/pytorch/default/1/330k_pearson_pruned.pt'
    ]

    MEANS = [(0.5212,0.4260,0.3811), (0.5207,0.4258,0.3806), (0.4868,0.3972,0.3624),
             (0.4668,0.3816,0.3414), (0.4923,0.4042,0.3624)]
    STDS = [(0.2486,0.2238,0.2211), (0.2490,0.2239,0.2212), (0.2296,0.2066,0.2009),
            (0.2410,0.2161,0.2081), (0.2446,0.2198,0.2141)]

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    # لود مدل‌ها
    base_models = load_pruned_models(MODEL_PATHS, device)

    # ساخت ensemble
    ensemble = FuzzyEnsembleModel(base_models, MEANS, STDS, freeze_models=True).to(device)
    if world_size > 1:
        ensemble = DDP(ensemble, device_ids=[rank])

    # دیتالودر
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, world_size, rank, num_workers=4)

    # فقط rank 0 آمار چاپ می‌کنه
    if rank == 0:
        total_params = sum(p.numel() for p in ensemble.module.parameters())
        trainable = sum(p.numel() for p in get_gating_network(ensemble.module).parameters())
        print(f"Total params: {total_params:,} | Trainable: {trainable:,} | Frozen: {total_params - trainable:,}\n")

    # آموزش
    best_val_acc, history = train_gating_network(
        ensemble, train_loader, val_loader,
        num_epochs=args.epochs, lr=args.lr, device=device,
        save_dir=args.save_dir, rank=rank
    )

    # فقط rank 0 ارزیابی نهایی
    if rank == 0:
        best_ckpt = torch.load(os.path.join(args.save_dir, 'best_fuzzy_gating.pt'), map_location=device)
        get_gating_network(ensemble.module).load_state_dict(best_ckpt['gating_state_dict'])
        print("Best gating network loaded.\n")

        print("="*70)
        val_acc, ind_accs_val, weights_val = evaluate_ensemble_detailed(
            ensemble.module, val_loader, device, "Validation", rank)
        print("="*70)
        test_acc, ind_accs_test, weights_test = evaluate_ensemble_detailed(
            ensemble.module, test_loader, device, "Test", rank)

        final_path = '/kaggle/working/fuzzy_ensemble_final_ddp.pt'
        torch.save({
            'gating_state_dict': get_gating_network(ensemble.module).state_dict(),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'individual_accs_val': ind_accs_val,
            'individual_accs_test': ind_accs_test,
            'weights_val': weights_val.tolist(),
            'weights_test': weights_test.tolist(),
            'history': history
        }, final_path)
        print(f"\nFinal model saved: {final_path}")
        print("All done!")

    # پاکسازی
    if world_size > 1:
        dist.destroy_process_group()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Fuzzy Gating Network with DDP")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size PER GPU')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--world_size', type=int, default=2, help='Number of GPUs')

    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
