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
    torch.cuda.manual_seed_all(seed)
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

class SimpleHead(nn.Module):
    def __init__(self, input_dim): # input_dim is the size of the concatenated features from the base models
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), # Added BatchNorm for stability
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128), # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1) # Output for binary classification
            # No Sigmoid here, as BCEWithLogitsLoss is used
        )

    def forward(self, x):
        return self.classifier(x)
# <<< MODIFICATION END

# <<< MODIFICATION START
# The FuzzyHesitantEnsemble is now a simple feature extractor + classifier.
# It concatenates features from frozen base models and passes them to SimpleHead.
class FuzzyHesitantEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], input_dim: int, freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        
        # We still need normalization for the base models
        class ModelNormalization(nn.Module):
            def __init__(self, means, stds):
                super().__init__()
                for i, (m, s) in enumerate(zip(means, stds)):
                    self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, -1, 1, 1))
                    self.register_buffer(f'std_{i}', torch.tensor(s).view(1, -1, 1, 1))
            def forward(self, x, idx):
                return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

        self.normalizations = ModelNormalization(means, stds)
        
        # The new classifier head
        self.classifier = SimpleHead(input_dim)
      
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
  
    def forward(self, x: torch.Tensor):
        # Collect features from all base models
        features_list = []
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad(): # Ensure gradients don't flow to frozen models
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            features_list.append(out)
      
        # Concatenate features and pass to the classifier
        concatenated_features = torch.cat(features_list, dim=1)
        final_output = self.classifier(concatenated_features)
        return final_output
# <<< MODIFICATION END
    
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
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == 'train'),
            drop_last=(split == 'train')
        )
      
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn
        )
        loaders[split] = loader
  
    if rank == 0:
        print(f"DataLoaders ready! Batch size per GPU: {batch_size}")
        print(f" Effective batch size: {batch_size * world_size}")
        print(f" Batches per GPU → Train: {len(loaders['train'])}, Val: {len(loaders['valid'])}, Test: {len(loaders['test'])}")
        print("="*70 + "\n")
    return loaders['train'], loaders['valid'], loaders['test']

# <<< MODIFICATION START
# The training function is simplified to train the new classifier head.
def train_simple_head_ddp(ensemble_model, train_loader, val_loader, num_epochs, lr,
                          device, save_dir, rank, world_size):
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    
    # The trainable part is now the 'classifier' head
    classifier_net = ensemble_model.module.classifier
    optimizer = torch.optim.AdamW(classifier_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
  
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
  
    if rank == 0:
        print("="*70)
        print("Training Simple Head Classifier (DDP)")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in classifier_net.parameters()):,}")
        print(f"World Size (GPUs): {world_size}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr}\n")
  
    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loader.sampler.set_epoch(epoch)
    
        train_loss = train_correct = train_total = 0.0
    
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]') if rank == 0 else train_loader
      
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device).float()
        
            optimizer.zero_grad()
            outputs = ensemble_model(images)
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
                iterator.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
      
        # Collect metrics from all GPUs
        train_loss_tensor = torch.tensor(train_loss, dtype=torch.float32).to(device)
        train_correct_tensor = torch.tensor(train_correct, dtype=torch.float32).to(device)
        train_total_tensor = torch.tensor(train_total, dtype=torch.float32).to(device)
    
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
    
        train_acc = 100. * train_correct_tensor.item() / train_total_tensor.item()
        train_loss = train_loss_tensor.item() / train_total_tensor.item()
      
        # Evaluate on validation set
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device, rank, world_size)
        scheduler.step(val_acc)
    
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
          
            print(f"\nEpoch {epoch+1}:")
            print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f" Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
          
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                tmp_path = os.path.join(save_dir, 'best_tmp.pt')
                final_path = os.path.join(save_dir, 'best_simple_head.pt')
              
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'classifier_state_dict': classifier_net.state_dict(),
                        'val_acc': val_acc,
                        'history': history
                    }, tmp_path)
                  
                    if os.path.exists(tmp_path):
                        shutil.move(tmp_path, final_path)
                        print(f" ✓ Best model saved → {val_acc:.2f}%")
                except Exception as e:
                    print(f" [ERROR] Failed to save model: {e}")
          
            print("-" * 70)
      
        dist.barrier()
  
    if rank == 0:
        print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc, history
# <<< MODIFICATION END

# <<< MODIFICATION START
# The evaluation functions are simplified as they no longer need to handle weights/memberships.
@torch.no_grad()
def evaluate_accuracy_ddp(model, loader, device, rank, world_size):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs = model(images) # Simple forward pass
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
  
    correct_tensor = torch.tensor(correct, dtype=torch.float32).to(device)
    total_tensor = torch.tensor(total, dtype=torch.float32).to(device)
  
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
  
    acc = 100. * correct_tensor.item() / total_tensor.item()
    return acc

def evaluate_final_ddp(model, loader, device, name, rank, world_size):
    model.eval()
    all_preds = []
    all_labels = []
  
    iterator = tqdm(loader, desc=f"Evaluating {name}") if rank == 0 else loader
  
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        pred = (outputs.squeeze(1) > 0).long()
      
        all_preds.append(pred)
        all_labels.append(labels)
  
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
  
    gathered_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
  
    dist.all_gather(gathered_preds, all_preds)
    dist.all_gather(gathered_labels, all_labels)
  
    acc = None
    if rank == 0:
        all_preds_final = torch.cat(gathered_preds).cpu().numpy()
        all_labels_final = torch.cat(gathered_labels).cpu().numpy()
        acc = 100. * np.mean(all_preds_final == all_labels_final)
        print(f"\n{name} Results → Accuracy: {acc:.2f}%")
  
    dist.barrier()
    return acc
# <<< MODIFICATION END

def main():
    SEED = 42
    set_seed(SEED)
  
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)
  
    parser = argparse.ArgumentParser(description="Train Simple Head Ensemble with DDP")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001) # Recommended learning rate
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                        help='List of paths to the pruned model files.')
  
    args = parser.parse_args()
  
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
  
    MODEL_NAMES = ["140k_pearson", "190k_pearson", "200k_kdfs"]
    MEANS = [(0.5207,0.4258,0.3806), (0.4868,0.3972,0.3624), (0.4668,0.3816,0.3414)]
    STDS = [(0.2490,0.2239,0.2212), (0.2296,0.2066,0.2009), (0.2410,0.2161,0.2081)]
    
    num_models_provided = len(args.model_paths)
    if num_models_provided > len(MODEL_NAMES):
        if is_main:
            print(f"[ERROR] Too many model paths provided ({num_models_provided}). Maximum supported is {len(MODEL_NAMES)}.")
        cleanup_ddp()
        return

    base_models = load_pruned_models(args.model_paths, device, rank)
    
    if len(base_models) != num_models_provided:
        if is_main:
            print(f"[WARNING] Only {len(base_models)}/{num_models_provided} models loaded successfully. Adjusting metadata lists.")
        MEANS = MEANS[:len(base_models)]
        STDS = STDS[:len(base_models)]
        MODEL_NAMES = MODEL_NAMES[:len(base_models)]

    # <<< MODIFICATION START
    # Dynamically determine the input dimension for the SimpleHead
    if is_main:
        print("Determining feature dimension for the new classifier head...")
    with torch.no_grad():
        dummy_input = torch.randn(args.batch_size, 3, 256, 256).to(device)
        dummy_output = base_models[0](dummy_input)
        if isinstance(dummy_output, (tuple, list)):
            out_tensor = dummy_output[0]
        else:
            out_tensor = dummy_output
        
        if out_tensor.dim() != 2:
             raise ValueError(f"Unexpected model output dimension: {out_tensor.shape}. Expected a 2D tensor [batch, features].")
        
        feature_dim_per_model = out_tensor.shape[1]
        input_dim = feature_dim_per_model * len(base_models)
        if is_main:
            print(f"Detected feature dimension per model: {feature_dim_per_model}")
            print(f"Input dimension for SimpleHead: {input_dim}\n")
    # <<< MODIFICATION END
  
    # <<< MODIFICATION START
    # Initialize the new ensemble model
    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        input_dim=input_dim, # Pass the calculated input_dim
        freeze_models=True
    ).to(device)
    # <<< MODIFICATION END

    ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if is_main:
        classifier_net = ensemble.module.classifier
        trainable = sum(p.numel() for p in classifier_net.parameters())
        total_params = sum(p.numel() for p in ensemble.parameters())
        print(f"Total params: {total_params:,} | Trainable: {trainable:,} | Frozen: {total_params - trainable:,}\n")
  
    train_loader, val_loader, test_loader = create_dataloaders_ddp(
        args.data_dir, args.batch_size, rank, world_size, args.num_workers
    )
    dist.barrier()
  
    # <<< MODIFICATION START
    # Train the new model
    best_val_acc, history = train_simple_head_ddp(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, rank, world_size
    )
    # <<< MODIFICATION END

    # Load the best model
    # <<< MODIFICATION START
    ckpt_path = os.path.join(args.save_dir, 'best_simple_head.pt')
    # <<< MODIFICATION END
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # <<< MODIFICATION START
        ensemble.module.classifier.load_state_dict(ckpt['classifier_state_dict'])
        # <<< MODIFICATION END
        if is_main:
            print("✓ Best simple head model loaded.\n")
    dist.barrier()
    if is_main:
        print("\n" + "="*70)
        print("EVALUATING ENSEMBLE WITH SIMPLE HEAD")
        print("="*70)
  
    # <<< MODIFICATION START
    test_acc = evaluate_final_ddp(
        ensemble, test_loader, device, "Test", rank, world_size
    )
    # <<< MODIFICATION END
  
    if is_main:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Ensemble with Simple Head Test Acc : {test_acc:.2f}%")
      
        results = {
            "method": "Simple Head Ensemble (DDP)",
            "seed": args.seed,
            "num_gpus": world_size,
            "test_acc": test_acc,
            "training_history": history
        }
      
        result_path = '/kaggle/working/simple_head_ddp_results.json'
        try:
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {result_path}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save results: {e}")
    
        final_model_path = '/kaggle/working/simple_head_ddp_final.pt'
        try:
            torch.save({
                # <<< MODIFICATION START
                'classifier_state_dict': ensemble.module.classifier.state_dict(),
                # <<< MODIFICATION END
                'results': results,
                'model_names': MODEL_NAMES,
                'means': MEANS,
                'stds': STDS
            }, final_model_path)
            print(f"✓ Final model saved: {final_model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save final model: {e}")
      
        print("\n" + "="*70)
        print("ALL DONE!")
        print("="*70)
  
    dist.barrier()
    cleanup_ddp()

if __name__ == "__main__":
    main()
