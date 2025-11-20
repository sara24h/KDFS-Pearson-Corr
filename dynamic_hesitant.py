import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
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
from sklearn.model_selection import train_test_split

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

# ====================== REAL_AND_FAKE_FACE SPLITTING ======================
def create_real_and_fake_face_splits(base_dir: str, seed: int = 42):
    real_dir = os.path.join(base_dir, 'training_real')
    fake_dir = os.path.join(base_dir, 'training_fake')
    if not (os.path.exists(real_dir) and os.path.exists(fake_dir)):
        raise FileNotFoundError(f"Expected 'training_real' and 'training_fake' in {base_dir}")

    # Use a dummy transform for initial loading
    dummy_transform = transforms.ToTensor()
    full_dataset = datasets.ImageFolder(base_dir, transform=dummy_transform)

    # Identify class indices
    real_class = full_dataset.class_to_idx['training_real']
    fake_class = full_dataset.class_to_idx['training_fake']

    real_indices = [i for i, (_, label) in enumerate(full_dataset.imgs) if label == real_class]
    fake_indices = [i for i, (_, label) in enumerate(full_dataset.imgs) if label == fake_class]

    def split(indices):
        train, temp = train_test_split(indices, test_size=0.3, random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed)
        random.seed(seed)
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        return train, val, test

    train_r, val_r, test_r = split(real_indices)
    train_f, val_f, test_f = split(fake_indices)

    train_indices = train_r + train_f
    val_indices = val_r + val_f
    test_indices = test_r + test_f

    random.seed(seed)
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)

    # Fix labels and class names
    def fix_labels(subset):
        new_samples = []
        for idx in subset.indices:
            path, _ = subset.dataset.samples[idx]
            label = 0 if 'training_real' in path else 1
            new_samples.append((path, label))
        subset.dataset.samples = new_samples
        subset.dataset.imgs = new_samples
        subset.dataset.classes = ['real', 'fake']
        subset.dataset.class_to_idx = {'real': 0, 'fake': 1}
        return subset

    train_set = fix_labels(train_set)
    val_set = fix_labels(val_set)
    test_set = fix_labels(test_set)

    return train_set, val_set, test_set

# ====================== DATALOADERS ======================
def create_dataloaders_ddp(base_dir: str, batch_size: int, rank: int, world_size: int,
                          num_workers: int = 2, dataset_name: str = 'wild20k', seed: int = 42):
    if rank == 0:
        print("="*70)
        print("Creating DataLoaders with DDP...")
        print("="*70)

    if dataset_name == 'wild20k':
        img_size = 256
        splits = ['train', 'valid', 'test']
        datasets_dict = {}
        for split in splits:
            path = os.path.join(base_dir, split)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Folder not found: {path}")
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5) if split == 'train' else transforms.Lambda(lambda x: x),
                transforms.RandomRotation(10) if split == 'train' else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(0.2, 0.2) if split == 'train' else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ])
            datasets_dict[split] = datasets.ImageFolder(path, transform=transform)

    elif dataset_name == 'real_and_fake_face':
        img_size = 600
        train_set, val_set, test_set = create_real_and_fake_face_splits(base_dir, seed=seed)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        train_set.dataset.transform = train_transform
        val_set.dataset.transform = val_test_transform
        test_set.dataset.transform = val_test_transform

        datasets_dict = {'train': train_set, 'valid': val_set, 'test': test_set}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if rank == 0:
        print(f"Dataset: {dataset_name} | Image size: {img_size}x{img_size}")
        print(f"Dataset Stats:")
        for split, ds in datasets_dict.items():
            if hasattr(ds, 'dataset'):
                num_samples = len(ds)
                classes = ['real', 'fake']
            else:
                num_samples = len(ds)
                classes = ds.classes
            print(f" {split.capitalize():5}: {num_samples:,} images | Classes: {classes}")

    loaders = {}
    for split, ds in datasets_dict.items():
        if split == 'train':
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
            loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init_fn)
        else:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False,
                              worker_init_fn=worker_init_fn)
        loaders[split] = loader

    if rank == 0:
        print(f"\nDataLoaders ready! Batch size per GPU: {batch_size}")
        print(f" Effective batch size: {batch_size * world_size}")
        print(f" Batches per GPU → Train: {len(loaders['train'])}, Val: {len(loaders['valid'])}, Test: {len(loaders['test'])}")
        print("="*70 + "\n")

    return loaders['train'], loaders['valid'], loaders['test']

# ====================== مدل‌ها و سایر توابع (بدون تغییر) ======================
# (کدهای HesitantFuzzyMembership, MultiModelNormalization, FuzzyHesitantEnsemble,
# load_pruned_models, evaluate_single_model, train_hesitant_fuzzy_ddp,
# evaluate_ensemble_final_ddp, evaluate_accuracy_ddp — همان‌طور که بود را حفظ کنید)

# برای اختصار، این بخش‌ها را کپی نمی‌کنم، اما شما **کل کد قبلی را حفظ کنید** و فقط
# تابع create_dataloaders_ddp و تابع جدید create_real_and_fake_face_splits را جایگزین کنید.
# همچنین main() را اصلاح کنید (در ادامه).

# ... [کل کلاس‌ها و توابع قبلی HesitantFuzzyMembership تا evaluate_accuracy_ddp را اینجا بیاورید] ...

# ====================== MAIN ======================
def main():
    SEED = 42
    set_seed(SEED)
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)

    parser = argparse.ArgumentParser(description="Train Fuzzy Hesitant Ensemble with DDP")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num_memberships', type=int, default=3)
    parser.add_argument('--dataset_name', type=str, default='wild20k',
                        choices=['wild20k', 'real_and_fake_face'],
                        help='Dataset to use')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    args = parser.parse_args()

    if args.seed != SEED:
        set_seed(args.seed)

    # تعیین مسیر دیتاست
    if args.dataset_name == 'wild20k':
        data_dir = '/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k'
    elif args.dataset_name == 'real_and_fake_face':
        data_dir = '/kaggle/input/real-and-fake-face'  # ✏️ مسیر خود را اینجا تنظیم کنید
    else:
        raise ValueError("Invalid dataset name")

    if is_main:
        print(f"="*70)
        print(f"Multi-GPU Training with DDP | SEED: {args.seed}")
        print(f"Dataset: {args.dataset_name} | Path: {data_dir}")
        print(f"="*70)
        print(f"World Size: {world_size} GPUs")
        print(f"Rank: {rank} | Local Rank: {local_rank} | Device: {device}")
        print(f"Batch size per GPU: {args.batch_size} | Effective batch size: {args.batch_size * world_size}")
        print(f"="*70 + "\n")

    # مدل‌ها (بدون تغییر)
    MODEL_PATHS = [
        '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
        '/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
        '/kaggle/input/200k-pearson-pruned/pytorch/default/1/200k_kdfs_pruned.pt',
    ]
    MODEL_NAMES = ["140k_pearson", "190k_pearson", "200k_kdfs"]
    MEANS = [(0.5207,0.4258,0.3806), (0.4868,0.3972,0.3624), (0.4668,0.3816,0.3414)]
    STDS = [(0.2490,0.2239,0.2212), (0.2296,0.2066,0.2009), (0.2410,0.2161,0.2081)]

    base_models = load_pruned_models(MODEL_PATHS, device, rank)
    if len(base_models) != len(MODEL_PATHS):
        if is_main:
            print(f"[WARNING] Only {len(base_models)}/{len(MODEL_PATHS)} models loaded.")
        MEANS = MEANS[:len(base_models)]
        STDS = STDS[:len(base_models)]
        MODEL_NAMES = MODEL_NAMES[:len(base_models)]

    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=0.9,
        hesitancy_threshold=0.2
    ).to(device)
    ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if is_main:
        hesitant_net = ensemble.module.hesitant_fuzzy
        trainable = sum(p.numel() for p in hesitant_net.parameters())
        total_params = sum(p.numel() for p in ensemble.parameters())
        print(f"Total params: {total_params:,} | Trainable: {trainable:,} | Frozen: {total_params - trainable:,}\n")

    # ایجاد دیتالودرها با پشتیبانی از دو دیتاست
    train_loader, val_loader, test_loader = create_dataloaders_ddp(
        data_dir, args.batch_size, rank, world_size,
        dataset_name=args.dataset_name,
        seed=args.seed
    )

    # ارزیابی مدل‌های فردی
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

        results = {
            "method": "Fuzzy Hesitant Sets (DDP)",
            "dataset": args.dataset_name,
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
