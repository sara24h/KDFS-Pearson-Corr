import json
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import argparse
from model.pruned_model.Resnet_final import ResNet_50_pruned_hardfakevsreal

class WildDeepfakeDataset(Dataset):
    def __init__(self, real_path, fake_path, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        if os.path.exists(real_path):
            real_files = [f for f in os.listdir(real_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in real_files:
                self.images.append(os.path.join(real_path, fname))
                self.labels.append(0)

        if os.path.exists(fake_path):
            fake_files = [f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in fake_files:
                self.images.append(os.path.join(fake_path, fname))
                self.labels.append(1)

        print(f"📊 Dataset loaded: {len(self.images)} images ({sum(1 for l in self.labels if l==0)} real, {sum(1 for l in self.labels if l==1)} fake)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float32)


# 🔥 IMPROVED: Data Augmentation قوی‌تر برای کاهش Overfitting
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4414, 0.3448, 0.3159], std=[0.1854, 0.1623, 0.1562]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4414, 0.3448, 0.3159], std=[0.1854, 0.1623, 0.1562])
])

def create_dataloaders(batch_size=256, num_workers=4):
    train_dataset = WildDeepfakeDataset(
        real_path="/kaggle/input/wild-deepfake/train/real",
        fake_path="/kaggle/input/wild-deepfake/train/fake",
        transform=train_transform
    )

    val_dataset = WildDeepfakeDataset(
        real_path="/kaggle/input/wild-deepfake/valid/real",
        fake_path="/kaggle/input/wild-deepfake/valid/fake",
        transform=val_transform
    )

    test_dataset = WildDeepfakeDataset(
        real_path="/kaggle/input/wild-deepfake/test/real",
        fake_path="/kaggle/input/wild-deepfake/test/fake",
        transform=val_transform
    )

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler

# 🔥 IMPROVED: Label Smoothing برای regularization بهتر
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(pred, target)

def train_epoch(model, loader, criterion, optimizer, device, scaler, writer, epoch, rank=0, accum_steps=1, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", disable=rank != 0)

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * accum_steps:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = torch.tensor(running_loss / len(loader)).to(device)
    avg_acc = torch.tensor(100. * correct / total).to(device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    avg_acc = avg_acc.item() / dist.get_world_size()

    if rank == 0 and writer is not None:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/acc", avg_acc, epoch)

    return avg_loss, avg_acc

@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch, rank=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Validation", disable=rank != 0):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = torch.tensor(running_loss / len(loader)).to(device)
    avg_acc = torch.tensor(100. * correct / total).to(device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    avg_acc = avg_acc.item() / dist.get_world_size()

    if rank == 0 and writer is not None:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/acc", avg_acc, epoch)

    return avg_loss, avg_acc

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
        return self.early_stop

def setup_ddp(seed):
    os.environ['TORCH_NCCL_TIMEOUT_MS'] = '1800000'
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    seed = seed + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def main(args):
    SEED = 42
    local_rank = setup_ddp(SEED)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    DEVICE = torch.device(f"cuda:{local_rank}")
    BATCH_SIZE_PER_GPU = args.batch_size
    BATCH_SIZE = BATCH_SIZE_PER_GPU * world_size
    NUM_EPOCHS = args.num_epochs
    BASE_LR = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    ACCUM_STEPS = args.accum_steps

    result_dir = f'/kaggle/working/runs_ddp_rank_{global_rank}'
    if global_rank == 0:
        writer = SummaryWriter(result_dir)
    else:
        writer = None

    if global_rank == 0:
        print("="*70)
        print("🚀 شروع Fine-tuning مدل Pruned ResNet50 — Layer3 + Layer4 + FC")
        print(f"   تعداد گرافیک: {world_size}")
        print(f"   Batch Size کل: {BATCH_SIZE}")
        print(f"   Gradient Accumulation Steps: {ACCUM_STEPS}")
        print(f"   Effective Batch Size: {BATCH_SIZE * ACCUM_STEPS}")
        print(f"   تعداد Epochs: {NUM_EPOCHS}")
        print(f"   Learning Rate: {BASE_LR}")
        print(f"   Weight Decay: {WEIGHT_DECAY}")
        print("="*70)

    if global_rank == 0:
        print("\n📦 لود مدل Pruned...")

    input_model_path = '/kaggle/input/330k-fuzzy-ranked-based-ensemble-5/330k_final.pt'
    checkpoint = torch.load(input_model_path, map_location=DEVICE)

    masks_detached = [m.detach().clone() if m is not None else None for m in checkpoint['masks']]

    model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)

    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 🔥 IMPROVED: FC layer با Dropout بیشتر و Batch Normalization
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    model = model.to(DEVICE)

    # 🔥 IMPROVED: Progressive unfreezing strategy
    for param in model.parameters():
        param.requires_grad = False

    # Layer3: Fine-tune با LR خیلی کم (0.1x base)
    for param in model.layer3.parameters():
        param.requires_grad = True

    # Layer4: Fine-tune با LR متوسط (0.5x base)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # FC: Train از صفر با LR بالا (1.0x base)
    for param in model.fc.parameters():
        param.requires_grad = True

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if global_rank == 0:
        print(f"✅ مدل لود و تنظیم شد")
        print(f"   - تعداد کل پارامترها: {total_params:,}")
        print(f"   - تعداد پارامترهای قابل آموزش: {trainable_params:,}")
        print(f"   - لایه‌های قابل آموزش: layer3 + layer4 + fc")

    if global_rank == 0:
        print("\n📊 آماده‌سازی DataLoaders...")

    train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = create_dataloaders(
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=2
    )

    # 🔥 IMPROVED: Label Smoothing برای regularization
    criterion = LabelSmoothingBCELoss(smoothing=0.1)

    # 🔥 IMPROVED: Differential Learning Rates (خیلی مهم!)
    # Layer3: 0.1x (تغییرات خیلی کم - حفظ pretrained features)
    # Layer4: 0.5x (تغییرات متوسط)
    # FC: 1.0x (یادگیری کامل از صفر)
    optimizer = optim.AdamW([
        {'params': model.module.layer3.parameters(), 'lr': BASE_LR * 0.1, 'weight_decay': WEIGHT_DECAY * 3},
        {'params': model.module.layer4.parameters(), 'lr': BASE_LR * 0.5, 'weight_decay': WEIGHT_DECAY * 2},
        {'params': model.module.fc.parameters(),     'lr': BASE_LR * 1.0, 'weight_decay': WEIGHT_DECAY * 3}
    ])
    
    # 🔥 IMPROVED: CosineAnnealing scheduler به جای StepLR
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-7
    )
    
    scaler = GradScaler(enabled=True)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        if global_rank == 0:
            print(f"\n📍 Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"   LR (layer3): {optimizer.param_groups[0]['lr']:.7f}")
            print(f"   LR (layer4): {optimizer.param_groups[1]['lr']:.7f}")
            print(f"   LR (fc):    {optimizer.param_groups[2]['lr']:.7f}")
            print("-" * 70)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, writer, 
            epoch, global_rank, ACCUM_STEPS, scheduler=scheduler
        )
        if global_rank == 0:
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, writer, epoch, global_rank)
        if global_rank == 0:
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.module.state_dict(), '/kaggle/working/best_model_improved.pt')
                print(f"✅ بهترین مدل ذخیره شد با Val Acc: {val_acc:.2f}%")

        scheduler.step()

        if early_stopping(val_acc):
            if global_rank == 0:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                print(f"Best Val Acc: {best_val_acc:.2f}%")
            break

    if global_rank == 0:
        model.module.load_state_dict(torch.load('/kaggle/working/best_model_improved.pt'))
    
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, writer, NUM_EPOCHS, global_rank)

    if global_rank == 0:
        print("\n" + "="*70)
        print("🧪 تست نهایی و ذخیره مدل")
        print("="*70)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        model_inference = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])
        in_features = model_inference.fc.in_features
        model_inference.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        model_inference.load_state_dict(model.module.state_dict())
        model_inference = model_inference.to('cpu')
        model_inference.eval()

        total_params_inf = sum(p.numel() for p in model_inference.parameters())

        checkpoint_inference = {
            'model_state_dict': model_inference.state_dict(),
            'total_params': total_params_inf,
            'masks': checkpoint['masks'],
            'model_architecture': 'ResNet_50_pruned_hardfakevsreal (Improved)',
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'training_config': {
                'lr': BASE_LR,
                'weight_decay': WEIGHT_DECAY,
                'batch_size': BATCH_SIZE,
                'accum_steps': ACCUM_STEPS,
                'epochs': NUM_EPOCHS,
                'loss': 'LabelSmoothingBCE',
                'dropout': [0.6, 0.5],
                'scheduler': 'CosineAnnealingWarmRestarts'
            }
        }

        inference_save_path = '/kaggle/working/final_improved_model.pt'
        torch.save(checkpoint_inference, inference_save_path)

        print("✅ مدل بهبودیافته با موفقیت ذخیره شد!")
        print(f"تعداد پارامترها: {total_params_inf:,}")
        print(f"بهترین Val Acc: {best_val_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%")

        file_size_mb = os.path.getsize(inference_save_path) / (1024 * 1024)
        print(f"حجم فایل: {file_size_mb:.2f} MB")

        writer.close()

    cleanup_ddp()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Fine-tune for WildDeepfake")
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU (256 recommended)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Base learning rate (1e-4 recommended)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps (1 recommended)')
    args = parser.parse_args()
    main(args)
