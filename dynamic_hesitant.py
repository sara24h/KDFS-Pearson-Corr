import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any
import warnings
import argparse
import shutil
import json
import random

warnings.filterwarnings("ignore")
# ====================== SEED SETUP ======================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for all GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[SEED] All random seeds set to: {seed}")
  
def worker_init_fn(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# =======================================================
def setup_ddp():
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size
  
def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
class HesitantFuzzyMembership(nn.Module):
    def __init__(self, input_dim: int, num_models: int, num_memberships: int = 3, dropout: float = 0.3):
        super().__init__()
        self.num_models = num_models
        self.num_memberships = num_memberships
      
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
      
        self.membership_generator = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_models * num_memberships)
        )
        self.aggregation_weights = nn.Parameter(torch.ones(num_memberships) / num_memberships)
      
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(x).flatten(1)
        memberships = self.membership_generator(features)
        memberships = memberships.view(-1, self.num_models, self.num_memberships)
        memberships = torch.sigmoid(memberships)
        agg_weights = F.softmax(self.aggregation_weights, dim=0)
        final_weights = (memberships * agg_weights.view(1, 1, -1)).sum(dim=2)
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
                 stds: List[Tuple[float]], num_memberships: int = 3, freeze_models: bool = True,
                 cum_weight_threshold: float = 0.9, hesitancy_threshold: float = 0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128,
            num_models=self.num_models,
            num_memberships=num_memberships
        )
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
    def forward(self, x: torch.Tensor, return_details: bool = False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)  # shapes: (batch_size, num_models), (batch_size, num_models, num_memberships)
        
        # calculate hesitancy (variance per model per sample)
        hesitancy = all_memberships.var(dim=2)  # shape: (batch_size, num_models)
        avg_hesitancy = hesitancy.mean(dim=1)  # average hesitancy per sample: (batch_size,)
        
        # initial mask: all weights active
        mask = torch.ones_like(final_weights)  # 1 => active, 0 => inactive
        
        # for samples with high hesitancy, keep all models active
        high_hesitancy_mask = (avg_hesitancy > self.hesitancy_threshold).unsqueeze(1)  # (batch_size, 1)
        # if high hesitancy, keep mask as all ones
        
        # for others: dynamic selection based on cumulative weight
        sorted_weights, sorted_indices = torch.sort(final_weights, dim=1, descending=True)  # (batch_size, num_models)
        
        cum_weights = torch.cumsum(sorted_weights, dim=1)  # cumulative sum (descending)
        
        # find minimal number of models to reach cum_threshold (per sample)
        # e.g., where cum_weights >= threshold -> set the rest to mask=0
        for b in range(x.size(0)):  # loop per-batch (usually small batch, low overhead)
            if high_hesitancy_mask[b]:  # if high hesitancy, keep all active
                continue
            # find first index where cum >= threshold
            active_count = torch.sum(cum_weights[b] < self.cum_weight_threshold) + 1
            # zero out models beyond top-active_count
            top_indices = sorted_indices[b, :active_count]
            sample_mask = torch.zeros(self.num_models, device=x.device)
            sample_mask[top_indices] = 1.0
            mask[b] = sample_mask
        
        # apply mask to weights
        final_weights = final_weights * mask
        
        # re-normalize (only active weights)
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # compute outputs only for active models (weight > 0)
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)  # assume binary class
        
        active_model_indices = set()  # to avoid running models that no sample needs
        for b in range(x.size(0)):
            active_model_indices.update(torch.nonzero(final_weights[b] > 0).squeeze(-1).cpu().tolist())
        
        for i in list(active_model_indices):
            # normalize and forward only for this model
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out  # fill outputs
        
        # aggregate
        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
        
        if return_details:
            return final_output, final_weights, all_memberships, outputs
        return final_output, final_weights
      
def load_pruned_models(model_paths: List[str], device: torch.device, rank: int) -> List[nn.Module]:
    try:
        from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal. Ensure model.pruned_model.ResNet_pruned is available.")
    models = []
    if rank == 0:
        print(f"Loading {len(model_paths)} pruned models...")
  
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if rank == 0:
                print(f" [WARNING] File not found: {path}")
            continue
      
        if rank == 0:
            print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
      
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
          
            if rank == 0:
                param_count = sum(p.numel() for p in model.parameters())
                print(f" → Parameters: {param_count:,}")
          
            models.append(model)
        except Exception as e:
            if rank == 0:
                print(f" [ERROR] Failed to load {path}: {e}")
            continue
  
    if len(models) == 0:
        raise ValueError("No models loaded!")
  
    if rank == 0:
        print(f"All {len(models)} models loaded!\n")
  
    return models
def create_dataloaders_ddp(base_dir: str, batch_size: int, rank: int, world_size: int, num_workers: int = 2):
    if rank == 0:
        print("="*70)
        print("Creating DataLoaders with DDP...")
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
      
        if rank == 0:
            print(f"{split.capitalize():5}: {path}")
      
        transform = train_transform if split == 'train' else val_test_transform
        datasets_dict[split] = datasets.ImageFolder(path, transform=transform)
    if rank == 0:
        print(f"\nDataset Stats:")
        for split, ds in datasets_dict.items():
            print(f" {split.capitalize():5}: {len(ds):,} images | Classes: {ds.classes}")
        print(f" Class → Index: {datasets_dict['train'].class_to_idx}\n")
    loaders = {}
    for split, ds in datasets_dict.items():
        if split == 'train':
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
            loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init_fn) # <--- SEED FOR WORKERS
        else:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False,
                              worker_init_fn=worker_init_fn) # <--- SEED FOR WORKERS
        loaders[split] = loader
    if rank == 0:
        print(f"DataLoaders ready! Batch size per GPU: {batch_size}")
        print(f" Effective batch size: {batch_size * world_size}")
        print(f" Batches per GPU → Train: {len(loaders['train'])}, Val: {len(loaders['valid'])}, Test: {len(loaders['test'])}")
        print("="*70 + "\n")
  
    return loaders['train'], loaders['valid'], loaders['test']

@torch.no_grad()
def evaluate_single_model_ddp(model, loader, device, name, rank, world_size):
  
    model.eval()
    correct = total = 0
    
    iterator = tqdm(loader, desc=f"Evaluating {name}") if rank == 0 else loader
    
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device).float()
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()

    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    acc = 100. * correct_tensor.item() / total_tensor.item()
    
    if rank == 0:
        print(f" {name}: {acc:.2f}% (total: {total_tensor.item()} samples)")
    
    return acc
  
def train_hesitant_fuzzy_ddp(ensemble_model, train_loader, val_loader, num_epochs, lr, device, save_dir, rank, world_size):
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
  
    hesitant_net = ensemble_model.module.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'membership_variance': []}
    if rank == 0:
        print("="*70)
        print("Training Fuzzy Hesitant Network (DDP)")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
        print(f"World Size (GPUs): {world_size}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr}")
        print(f"Hesitant memberships per model: {hesitant_net.num_memberships}\n")
    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loader.sampler.set_epoch(epoch)
      
        train_loss = train_correct = train_total = 0.0
        membership_vars = []
      
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]') if rank == 0 else train_loader
        for images, labels in iterator:
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
          
            membership_vars.append(memberships.var(dim=2).mean().item())
            if rank == 0:
                current_acc = 100. * train_correct / train_total
                avg_loss = train_loss / train_total
                iterator.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{current_acc:.2f}% '})
        # gather metrics from all GPUs
        train_loss_tensor = torch.tensor(train_loss).to(device)
        train_correct_tensor = torch.tensor(train_correct).to(device)
        train_total_tensor = torch.tensor(train_total).to(device)
      
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
      
        train_acc = 100. * train_correct_tensor.item() / train_total_tensor.item()
        train_loss = train_loss_tensor.item() / train_total_tensor.item()
        avg_membership_var = np.mean(membership_vars)
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device, rank)
        scheduler.step()
      
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['membership_variance'].append(avg_membership_var)
            print(f"\nEpoch {epoch+1}:")
            print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f" Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f" Membership Variance (Hesitancy): {avg_membership_var:.4f}")
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
                        print(f" Best model saved → {val_acc:.2f}%")
                except Exception as e:
                    print(f" [ERROR] Failed to save model: {e}")
            print("-" * 70)
        dist.barrier()
    if rank == 0:
        print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
  
    return best_val_acc, history
    
@torch.no_grad()
def evaluate_ensemble_final_ddp_fixed(model, loader, device, name, model_names, rank, world_size):
    
    model.eval()
    all_preds, all_labels = [], []
    all_weights_list, all_memberships_list = [], []
    
    iterator = tqdm(loader, desc=f"Evaluating {name}") if rank == 0 else loader
    
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        outputs, weights, memberships, _ = model(images, return_details=True)
        pred = (outputs.squeeze(1) > 0).long()
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_weights_list.append(weights.cpu().numpy())
        all_memberships_list.append(memberships.cpu().numpy())
    
    # محاسبه دقت
    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    total = len(all_labels)
    
    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    acc = 100. * correct_tensor.item() / total_tensor.item()
    
    # محاسبه میانگین وزن‌ها
    local_avg_weights = np.concatenate(all_weights_list).mean(axis=0)
    local_avg_memberships = np.concatenate(all_memberships_list).mean(axis=0)
    
    weights_tensor = torch.tensor(local_avg_weights, device=device)
    memberships_tensor = torch.tensor(local_avg_memberships, device=device)
    
    dist.all_reduce(weights_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(memberships_tensor, op=dist.ReduceOp.SUM)
    
    avg_weights = (weights_tensor / world_size).cpu().numpy()
    avg_memberships = (memberships_tensor / world_size).cpu().numpy()
    
    if rank == 0:
        print(f"\n{name} Results → Accuracy: {acc:.2f}% (total: {total_tensor.item()} samples)")
        print("\nFinal Hesitant Fuzzy Weights:")
        for i, (w, name) in enumerate(zip(avg_weights, model_names)):
            print(f" Model {i+1} ({name}): {w:.4f} ({w*100:.2f}%)")
        
        print("\nHesitant Membership Values per Model:")
        for i, name in enumerate(model_names):
            memberships = avg_memberships[i]
            variance = memberships.var()
            print(f" Model {i+1} ({name}):")
            print(f"   Memberships: {[f'{m:.3f}' for m in memberships]}")
            print(f"   Variance (Hesitancy): {variance:.4f}")
    
    return acc, avg_weights.tolist(), avg_memberships.tolist()
    
@torch.no_grad()
def evaluate_accuracy_ddp(model, loader, device, rank):
    model.eval()
    correct = total = 0
  
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
  
    acc = 100. * correct / total
    return acc
  
def main():
    # ====================== SET SEED BEFORE DDP ======================
    SEED = 42
    set_seed(SEED)
    # Setup DDP
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
  
    is_main = (rank == 0)
    parser = argparse.ArgumentParser(description="Train Fuzzy Hesitant Ensemble with DDP")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num_memberships', type=int, default=3, help='Number of membership values per model')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed for reproducibility')
    args = parser.parse_args()
    # Override seed if provided via CLI
    if args.seed != SEED:
        set_seed(args.seed)
    if is_main:
        print(f"="*70)
        print(f"Multi-GPU Training with DDP | SEED: {args.seed}")
        print(f"="*70)
        print(f"World Size: {world_size} GPUs")
        print(f"Rank: {rank} | Local Rank: {local_rank} | Device: {device}")
        print(f"Batch size per GPU: {args.batch_size} | Effective batch size: {args.batch_size * world_size}")
        print(f"="*70 + "\n")
    # Models
    MODEL_PATHS = [
        '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
        '/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
        '/kaggle/input/200k-pearson-pruned/pytorch/default/1/200k_kdfs_pruned.pt',
    ]
  
    MODEL_NAMES = ["140k_pearson", "190k_pearson", "200k_pearson"]
    MEANS = [(0.5207,0.4258,0.3806), (0.4868,0.3972,0.3624), (0.4668,0.3816,0.3414)]
    STDS = [(0.2490,0.2239,0.2212), (0.2296,0.2066,0.2009), (0.2410,0.2161,0.2081)]
  
    base_models = load_pruned_models(MODEL_PATHS, device, rank)
  
    if len(base_models) != len(MODEL_PATHS):
        if is_main:
            print(f"[WARNING] Only {len(base_models)}/{len(MODEL_PATHS)} models loaded. Adjusting MEANS/STDS.")
        MEANS = MEANS[:len(base_models)]
        STDS = STDS[:len(base_models)]
        MODEL_NAMES = MODEL_NAMES[:len(base_models)]
    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=0.9,  # adjust this to control "top" models
        hesitancy_threshold=0.2    # adjust this for "use all models" cases
    ).to(device)
  
    ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
  
    if is_main:
        hesitant_net = ensemble.module.hesitant_fuzzy
        trainable = sum(p.numel() for p in hesitant_net.parameters())
        total_params = sum(p.numel() for p in ensemble.parameters())
        print(f"Total params: {total_params:,} | Trainable: {trainable:,} | Frozen: {total_params - trainable:,}\n")
    train_loader, val_loader, test_loader = create_dataloaders_ddp(
        args.data_dir, args.batch_size, rank, world_size
    )
  
    if is_main:
        print("\n" + "="*70)
        print("EVALUATING INDIVIDUAL MODELS ON TEST SET (Before Training)")
        print("="*70)
        individual_accs = []
        for i, model in enumerate(base_models):
            acc = evaluate_single_model(model, test_loader, device, f"Model {i+1} ({MODEL_NAMES[i]})", rank)
            individual_accs.append(acc)
        best_single = max(individual_accs)
        best_idx = individual_accs.index(best_single)
        print(f"\nBest Single Model: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
      
    dist.barrier()
    best_val_acc, history = train_hesitant_fuzzy_ddp(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, rank, world_size
    )
  
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
  
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ensemble.module.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        if is_main:
            print("Best hesitant fuzzy network loaded.\n")
  
    dist.barrier()
  
    if is_main:
        print("\n" + "="*70)
        print("EVALUATING FUZZY HESITANT ENSEMBLE")
        print("="*70)
        ensemble_test_acc, ensemble_weights, membership_values = evaluate_ensemble_final_ddp(
            ensemble, test_loader, device, "Test", MODEL_NAMES, rank
        )
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model Acc : {best_single:.2f}%")
        print(f"Hesitant Ensemble Acc : {ensemble_test_acc:.2f}%")
        improvement = ensemble_test_acc - best_single
        print(f"Improvement : {improvement:+.2f}%")
        # save results with seed
        results = {
            "method": "Fuzzy Hesitant Sets (DDP)",
            "seed": args.seed,
            "num_gpus": world_size,
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
        result_path = '/kaggle/working/hesitant_fuzzy_ddp_results.json'
      
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {result_path}")
      
        final_model_path = '/kaggle/working/hesitant_fuzzy_ddp_final.pt'
        torch.save({
            'hesitant_state_dict': ensemble.module.hesitant_fuzzy.state_dict(),
            'results': results
        }, final_model_path)
      
        print(f"Final model saved: {final_model_path}")
        print("All done!")
    cleanup_ddp()
  
if __name__ == "__main__":
    main()
