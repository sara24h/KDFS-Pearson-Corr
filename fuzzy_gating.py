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
import tempfile
import shutil

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
        weights = self.gating_network(x)  # (B, N)
        outputs = []

        for i, model in enumerate(self.models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = model(x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # (B, N, C)
        final_output = (outputs * weights.unsqueeze(-1)).sum(dim=1)

        if return_individual:
            return final_output, weights, outputs
        return final_output, weights


# =============================================================================
# تابع کمکی: دسترسی ایمن به gating_network
# =============================================================================

def get_gating_network(model: nn.Module) -> nn.Module:
    if hasattr(model, 'module'):
        return model.module.gating_network
    return model.gating_network


# =============================================================================
# لود مدل‌ها
# =============================================================================

def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

    models = []
    print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        print(f"  [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        ckpt = torch.load(path, map_location='cpu')
        model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device).eval()

        param_count = sum(p.numel() for p in model.parameters())
        print(f"     → Parameters: {param_count:,}")
        models.append(model)
    print(f"All {len(models)} models loaded successfully!\n")
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
    return datasets_dict['train'], datasets_dict['valid'], datasets_dict['test']


# =============================================================================
# ارزیابی با DDP
# =============================================================================

@torch.no_grad()
def evaluate_accuracy_ddp(model, loader, device, world_size):
    model.eval()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        total += labels.size(0)

    metrics = torch.tensor([correct, total], device=device, dtype=torch.float32)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return 100. * metrics[0].item() / metrics[1].item()


# =============================================================================
# آموزش Gating Network با DDP
# =============================================================================

def train_gating_network(
    ensemble_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    save_dir: str,
    local_rank: int,
    world_size: int
):
    os.makedirs(save_dir, exist_ok=True)
    gating_net = get_gating_network(ensemble_model)
    optimizer = torch.optim.AdamW(gating_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    if local_rank == 0:
        print("="*70)
        print("Training Fuzzy Gating Network (DDP + BCEWithLogitsLoss)")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in gating_net.parameters()):,}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr} | Device: {device}\n")

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        ensemble_model.train()
        train_loss = train_correct = train_total = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Rank {local_rank}]') if local_rank == 0 else train_loader

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

        # جمع‌آوری از همه GPUها
        metrics = torch.tensor([train_loss, train_correct, train_total], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        train_loss = metrics[0].item() / metrics[2].item()
        train_acc = 100. * metrics[1].item() / metrics[2].item()
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device, world_size)

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if local_rank == 0:
            print(f"\nEpoch {epoch+1}:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # ذخیره امن با tempfile
                tmp_path = os.path.join(save_dir, f'best_tmp_rank{local_rank}.pt')
                final_path = os.path.join(save_dir, 'best_fuzzy_gating.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'gating_state_dict': gating_net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': history
                }, tmp_path)
                shutil.move(tmp_path, final_path)
                print(f"   Best model saved → {val_acc:.2f}%")
            print("-" * 70)

    if local_rank == 0:
        print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, history


# =============================================================================
# ارزیابی نهایی
# =============================================================================

@torch.no_grad()
def evaluate_ensemble_final(model, loader, device, name, local_rank, world_size):
    model.eval()
    all_preds, all_labels, all_weights = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs, weights, _ = model(images, return_individual=True)
        pred = (outputs.squeeze(1) > 0).long()

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_weights.append(weights.cpu().numpy())

    # جمع‌آوری از همه GPUها
    preds_tensor = torch.tensor(all_preds, device=device)
    labels_tensor = torch.tensor(all_labels, device=device)
    weights_np = np.concatenate(all_weights)

    # all_gather برای لیست‌ها
    gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_preds, preds_tensor)
    dist.all_gather(gathered_labels, labels_tensor)

    all_preds = torch.cat(gathered_preds).cpu().numpy()
    all_labels = torch.cat(gathered_labels).cpu().numpy()

    # میانگین وزنی
    weights_list = [torch.zeros(weights_np.shape[0], 5, device=device) for _ in range(world_size)]
    weights_tensor = torch.tensor(weights_np, device=device)
    dist.all_gather(weights_list, weights_tensor)
    avg_weights = torch.cat(weights_list).mean(dim=0).cpu().numpy()

    acc = 100. * np.mean(all_preds == all_labels)

    if local_rank == 0:
        print("\n" + "="*70)
        print(f"{name} Results → Accuracy: {acc:.2f}%")
        print("Average Fuzzy Weights:")
        for i, w in enumerate(avg_weights):
            print(f"   Model {i+1}: {w:.4f} ({w*100:.2f}%)")
        print("="*70)
    return acc, avg_weights


# =============================================================================
# Main Worker
# =============================================================================

def main_worker(local_rank: int, world_size: int, args):
    # DDP init
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    def rank_print(*args, **kwargs):
        if local_rank == 0:
            print(*args, **kwargs)

    rank_print(f"Rank {local_rank} initialized. Device: {device}")

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
    seed = 42 + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # لود مدل‌ها
    base_models = load_pruned_models(MODEL_PATHS, device)

    # Ensemble
    ensemble = FuzzyEnsembleModel(base_models, MEANS, STDS, freeze_models=True)
    ensemble = ensemble.to(device)

    # فقط gating network آموزش داده بشه
    gating_net = get_gating_network(ensemble)
    gating_net = DDP(gating_net, device_ids=[local_rank], output_device=local_rank)

    # آمار
    total_params = sum(p.numel() for p in ensemble.parameters())
    trainable = sum(p.numel() for p in gating_net.parameters())
    rank_print(f"Total params: {total_params:,} | Trainable: {trainable:,} | Frozen: {total_params - trainable:,}\n")

    # دیتاست‌ها
    train_dataset, val_dataset, test_dataset = create_dataloaders(
        args.data_dir, args.batch_size, num_workers=2
    )

    # Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler, num_workers=2, pin_memory=True)

    rank_print(f"Batches → Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}\n")

    # آموزش
    best_val_acc, history = train_gating_network(
        ensemble, train_loader, val_loader,
        num_epochs=args.epochs, lr=args.lr, device=device,
        save_dir=args.save_dir, local_rank=local_rank, world_size=world_size
    )

    # بارگذاری بهترین مدل (فقط rank 0)
    if local_rank == 0:
        ckpt_path = os.path.join(args.save_dir, 'best_fuzzy_gating.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            gating_net.module.load_state_dict(ckpt['gating_state_dict'])
            print("Best gating network loaded.\n")
        else:
            print("Warning: Checkpoint not found! Using current state.")

    # ارزیابی نهایی
    val_acc, val_weights = evaluate_ensemble_final(ensemble, val_loader, device, "Validation", local_rank, world_size)
    test_acc, test_weights = evaluate_ensemble_final(ensemble, test_loader, device, "Test", local_rank, world_size)

    # ذخیره نهایی (فقط rank 0)
    if local_rank == 0:
        final_path = '/kaggle/working/fuzzy_ensemble_final_ddp.pt'
        torch.save({
            'gating_state_dict': gating_net.module.state_dict(),
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'val_weights': val_weights.tolist(),
            'test_weights': test_weights.tolist(),
            'history': history
        }, final_path)
        print(f"\nFinal DDP model saved: {final_path}")

    dist.destroy_process_group()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Fuzzy Gating Network with DDP")
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Error: Need at least 2 GPUs for DDP!")
        return

    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
