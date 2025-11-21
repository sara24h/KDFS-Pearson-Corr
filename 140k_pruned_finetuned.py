import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
import warnings
import random
import torch.optim.lr_scheduler as lr_scheduler 

# 💡 فرض بر این است که ایمپورت مدل واقعی موفق است و مسیر درست است.
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal 

warnings.filterwarnings("ignore")

# ====================== بارگذاری مدل (مدل واقعی برای Fine-Tuning) ======================
def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    """بارگذاری مدل‌های پایه هرس‌شده ResNet-50 و تنظیم برای Fine-Tuning."""
    
    if not model_paths:
        print("[ERROR] MODEL_PATHS cannot be empty for single model fine-tuning.")
        raise ValueError("No model path provided.")
        
    path = model_paths[0]
    
    masks = None
    try:
        # بارگذاری چک پوینت
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        masks = ckpt.get('masks')
        if masks is None:
            print(f"[INFO] 'masks' not found in {os.path.basename(path)}. Proceeding without masks.")
            
    except Exception as e:
        print(f"[ERROR] Could not load checkpoint from {path}: {e}")
        raise RuntimeError(f"Failed to load checkpoint for model initialization: {path}")

    # ⬅️ نمونه‌سازی مدل واقعی
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    # بارگذاری وزن‌های مدل
    model.load_state_dict(ckpt['model_state_dict'])
    
    # تنظیم مدل برای Fine-Tuning: انتقال به دیوایس و آماده‌سازی برای آموزش
    model = model.to(device)
    # 💡 توجه: مدل را در اینجا به .train() نمی‌بریم؛ در تابع fine_tune_single_model_executor این کار انجام می‌شود.
    # اما اگر مدل هرس‌شده شما در حالت eval() بهینه‌تر عمل می‌کند (برای جلوگیری از تغییر وزن‌های هرس)،
    # باید در تابع fine_tune بخش model.train() را به‌درستی مدیریت کنید.
    # ما آن را به حالت پیش‌فرض (آموزش) می‌گذاریم.
    
    print(f"Loaded 1 ResNet_50 model using checkpoint path: {path}")
    return [model]

# ====================== توابع کمکی (Single Process Setup) ======================
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

# ----------------------------------------------------------------------

# ====================== DataLoaders (Single Process) ======================
def create_dataloaders_single_gpu(base_dir: str, batch_size: int, num_workers: int = 2):
    """ایجاد DataLoaders استاندارد"""
    
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
    
    datasets_dict = {}
    for split in ['train', 'valid', 'test']:
        path = os.path.join(base_dir, split)
        if not os.path.exists(path):
             print(f"[ERROR] Folder not found: {path}")
             raise FileNotFoundError(f"Folder not found: {path}")
        
        transform = train_transform if split == 'train' else val_test_transform
        datasets_dict[split] = datasets.ImageFolder(path, transform=transform)
    
    loaders = {}
    for split, ds in datasets_dict.items():
        loader = DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=(split == 'train'), 
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=(split == 'train'),
            worker_init_fn=worker_init_fn
        )
        loaders[split] = loader
    
    print(f"DataLoaders ready! Batch size: {batch_size}. Train Samples: {len(datasets_dict['train']):,}")
    
    return loaders['train'], loaders['valid'], loaders['test']

# ----------------------------------------------------------------------

# ====================== ارزیابی تک مدل ======================
@torch.no_grad()
def evaluate_single_model_ft(model: nn.Module, loader: DataLoader, device: torch.device, 
                             name: str, mean: Tuple[float], std: Tuple[float]) -> float:
    """ارزیابی یک مدل در حالت تک فرآیند"""
    
    # 💡 برای ارزیابی حتماً مدل را در حالت eval قرار دهید
    model.eval() 
    correct = total = 0
    normalize = transforms.Normalize(mean=mean, std=std)
    
    iterator = tqdm(loader, desc=f"Evaluating {name}")
    
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device).float()
        images = normalize(images) 
        
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
        
    acc = 100. * correct / total
    print(f" {name} Accuracy: {acc:.2f}%")
        
    return acc

# ----------------------------------------------------------------------

# ====================== تابع اصلی Fine-Tuning (Single Process) ======================
def fine_tune_single_model_executor(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                                device: torch.device, ft_epochs: int, ft_lr: float, 
                                save_dir: str, mean: Tuple[float], std: Tuple[float], 
                                model_name: str) -> float:
    """Fine-tunes a single base model in a single process."""

    print(f"\n{'#'*70}")
    print(f"[{model_name}] Starting Fine-Tuning on {device}...")
    print(f" Epochs: {ft_epochs} | Learning Rate: {ft_lr} | Mean/Std: {mean}/{std}")
    trainable_params = sum(p.numel() for p in model.parameters())
    print(f" Total Trainable Params: {trainable_params:,}")
    print(f"{'#'*70}")
    
    # 💡 برای آموزش حتماً مدل را در حالت train قرار دهید
    model.train()
    # اطمینان از اینکه تمامی پارامترها قابل آموزش هستند
    for p in model.parameters():
        p.requires_grad = True

    normalize = transforms.Normalize(mean=mean, std=std)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs) 
    
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(ft_epochs):
        model.train()
        train_loss = train_correct = train_total = 0.0
        
        iterator = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1}/{ft_epochs} [Train]')
        
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device).float()
            images = normalize(images) 
            
            optimizer.zero_grad()
            out = model(images)
            
            if isinstance(out, (tuple, list)):
                out = out[0]
            
            loss = criterion(out.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pred = (out.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += images.size(0)
            
            current_acc = 100. * train_correct / train_total
            avg_loss = train_loss / train_total
            iterator.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{current_acc:.2f}%'})

        val_acc = evaluate_single_model_ft(model, val_loader, device, f"[{model_name}] Validation", mean, std)
        scheduler.step()
        
        final_train_acc = 100. * train_correct / train_total
        print(f"\n[{model_name}] Epoch {epoch+1} Summary:")
        print(f" Train Loss: {train_loss / train_total:.4f} | Train Acc: {final_train_acc:.2f}%")
        print(f" Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f'{model_name}_ft_best.pt')
            
            masks = getattr(model, 'masks', None) 
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'masks': masks, 
                'val_acc': val_acc,
            }, save_path)
            print(f" ✓ Best Fine-Tuned model saved to {save_path} → {val_acc:.2f}%")
        
    print(f"\n[{model_name}] Fine-Tuning Completed! Best Val Acc: {best_val_acc:.2f}%")
        
    model.eval()
    return best_val_acc

# ----------------------------------------------------------------------

# ====================== MAIN FUNCTION (تک مدل و Hardcoded) ======================
def main():

    FT_EPOCHS = 10
    FT_LR = 1e-4
    BATCH_SIZE = 64
    DATA_DIR = '/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k' 
    SAVE_DIR = '/kaggle/working/finetuned_models'
    NUM_WORKERS = 4
    
    SEED = 42
    set_seed(SEED)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to: {device}")
    
    print(f"="*70)
    print(f"Single Model Fine-Tuning | Device: {device}")
    print(f"Epochs: {FT_EPOCHS} | LR: {FT_LR} | Batch Size: {BATCH_SIZE}")
    print(f"="*70 + "\n")

    # ====================== DATA & MODEL DEFINITION (فقط یک مدل) ======================
    
    MODEL_PATHS = [
        '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
    ]
    MODEL_NAME = "140k_pearson" 
    MEAN = (0.5207,0.4258,0.3806)
    STD = (0.2490,0.2239,0.2212)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # بارگذاری مدل 
    base_models = load_pruned_models(MODEL_PATHS, device)
    model = base_models[0]

    # ====================== DATALOADERS ======================
    train_loader, val_loader, test_loader = create_dataloaders_single_gpu(
        DATA_DIR, BATCH_SIZE, NUM_WORKERS
    )
    
    # ====================== FINE-TUNING EXECUTION ======================
    print(f"Starting Fine-Tuning for Model: {MODEL_NAME}")
    
    best_val_acc = fine_tune_single_model_executor(
        model, train_loader, val_loader, 
        device, FT_EPOCHS, FT_LR, 
        SAVE_DIR, MEAN, STD, MODEL_NAME
    )
    
    final_accs = {MODEL_NAME: best_val_acc}

    # ====================== FINAL EVALUATION ======================
    print("\n" + "="*70)
    print(f"Final Test Evaluation ({MODEL_NAME})")
    print("="*70)
        
    best_ft_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_ft_best.pt')
    
    if os.path.exists(best_ft_path):
        try:
            # بارگذاری بهترین مدل ذخیره شده
            ckpt = torch.load(best_ft_path, map_location=device, weights_only=False)
            
            # مدل را با ماسک‌های درست (اگر وجود دارند) نمونه‌سازی می‌کنیم.
            model_eval = ResNet_50_pruned_hardfakevsreal(masks=ckpt.get('masks')).to(device).eval() 
            model_eval.load_state_dict(ckpt['model_state_dict'])
            
            # ارزیابی روی Test Set
            test_acc = evaluate_single_model_ft(model_eval, test_loader, device, 
                                                 f"{MODEL_NAME} Final Test", MEAN, STD)
            final_accs[f'{MODEL_NAME}_test'] = test_acc
        except Exception as e:
            print(f"[ERROR] Could not load or evaluate {MODEL_NAME}: {e}")
    else:
        print(f"[INFO] Best checkpoint not found for {MODEL_NAME}.")

    # ====================== RESULTS ======================
    print("\n" + "="*70)
    print("Final Fine-Tuning Results:")
    for name, acc in final_accs.items():
        print(f" {name}: {acc:.2f}%")
    print("ALL DONE!")
    print("="*70)

if __name__ == "__main__":
    main()
