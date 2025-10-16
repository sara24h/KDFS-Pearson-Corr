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

# ============================================================
# 1. تعریف Dataset سفارشی
# ============================================================
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

# ============================================================
# 2. تعریف Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4414, 0.3448, 0.3159], std=[0.1854, 0.1623, 0.1562])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4414, 0.3448, 0.3159], std=[0.1854, 0.1623, 0.1562])
])

# ============================================================
# 3. آماده‌سازی DataLoaders (با drop_last=True)
# ============================================================
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

    # ⚠️ drop_last=True برای جلوگیری از عدم تقارن در آخرین بچ
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler

# ============================================================
# 4. تابع آموزش
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler, writer, epoch, rank=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", disable=rank != 0)

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item()
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
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

# ============================================================
# 5. تابع setup DDP و seed
# ============================================================
def setup_ddp(seed):
    # ⚠️ افزایش تایم‌اوت NCCL برای محیط‌های کند (مثل Kaggle)
    os.environ['TORCH_NCCL_TIMEOUT_MS'] = '1800000'  # 30 دقیقه

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

# ============================================================
# 6. اصلی برنامه Fine-tuning
# ============================================================
def main():
    SEED = 42
    local_rank = setup_ddp(SEED)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    DEVICE = torch.device(f"cuda:{local_rank}")
    BATCH_SIZE_PER_GPU = 256
    BATCH_SIZE = BATCH_SIZE_PER_GPU * world_size
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4

    result_dir = f'/kaggle/working/runs_ddp_rank_{global_rank}'
    if global_rank == 0:
        writer = SummaryWriter(result_dir)
    else:
        writer = None

    if global_rank == 0:
        print("="*70)
        print("🚀 شروع Fine-tuning مدل Pruned ResNet50 با DDP و Mixed Precision")
        print(f"   تعداد گرافیک: {world_size}")
        print(f"   Batch Size کل: {BATCH_SIZE}")
        print("="*70)

    if global_rank == 0:
        print("\n📦 لود مدل Pruned...")

    input_model_path = '/kaggle/input/10k_final/pytorch/default/1/10k_final.pt'
    checkpoint = torch.load(input_model_path, map_location=DEVICE)

    masks_detached = [m.detach().clone() if m is not None else None for m in checkpoint['masks']]

    model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    # فریز کردن تمام لایه‌ها
    for param in model.parameters():
        param.requires_grad = False

    # باز کردن لایه آخر کانولوشنی (layer4) و لایه Fully Connected (fc)
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # محاسبه تعداد پارامترها
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if global_rank == 0:
        print(f"✅ مدل لود شد")
        print(f"   - تعداد کل پارامترها: {total_params:,}")
        print(f"   - تعداد پارامترهای قابل آموزش: {trainable_params:,}")
        print(f"   - لایه‌های قابل آموزش: layer4 و fc")

    if global_rank == 0:
        print("\n📊 آماده‌سازی DataLoaders...")

    train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = create_dataloaders(
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=2
    )

    criterion = nn.BCEWithLogitsLoss()
    # فقط پارامترهای قابل آموزش به optimizer اضافه می‌شوند
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = GradScaler(enabled=True)

    if global_rank == 0:
        print("\n" + "="*70)
        print("🎓 شروع آموزش")
        print("="*70)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        if global_rank == 0:
            print(f"\n📍 Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 70)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, writer, epoch, global_rank
        )
        if global_rank == 0:
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, writer, epoch, global_rank)
        if global_rank == 0:
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step()

    # تست نهایی
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, writer, NUM_EPOCHS, global_rank)

    if global_rank == 0:
        print("\n" + "="*70)
        print("🧪 تست نهایی و ذخیره مدل inference-ready")
        print("="*70)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        # بازسازی مدل روی CPU برای inference
        model_inference = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])
        model_inference.load_state_dict(model.module.state_dict())
        model_inference = model_inference.to('cpu')
        model_inference.eval()

        total_params_inf = sum(p.numel() for p in model_inference.parameters())

        # فقط 4 کلید مورد نظر
        checkpoint_inference = {
            'model_state_dict': model_inference.state_dict(),
            'total_params': total_params_inf,
            'masks': checkpoint['masks'],
            'model_architecture': 'ResNet_50_pruned_hardfakevsreal'
        }

        inference_save_path = '/kaggle/working/final_pruned_finetuned_inference_ready.pt'
        torch.save(checkpoint_inference, inference_save_path)

        # چاپ اطلاعات
        print("فایل شامل یک دیکشنری وزن‌ها است.")
        print("کلیدهای موجود در دیکشنری:")
        for key in checkpoint_inference.keys():
            print(f"- {key}")

        print("\nجزئیات وزن‌ها:")
        for key, value in checkpoint_inference.items():
            if key == 'masks':
                print(f"{key}: نوع = {type(value)} (list of {len(value)} masks)")
            else:
                print(f"{key}: نوع = {type(value)}")

        print("✅ مدل هرس‌شده با موفقیت بازسازی و لود شد!")
        print(f"تعداد پارامترها: {total_params_inf:,}")

        print("\n" + "="*70)
        print("معماری نهایی مدل هرس‌شده (ResNet_50_pruned_hardfakevsreal)")
        print("="*70)
        print(model_inference)
        print("\n" + "="*70)
        print("توجه: ابعاد هر لایه، معماری فشرده‌شده (هرس‌شده) را نشان می‌دهد.")
        print("="*70)

        file_size_mb = os.path.getsize(inference_save_path) / (1024 * 1024)
        print(f"✅ مدل inference-ready با موفقیت در {inference_save_path} ذخیره شد.")
        print(f"حجم فایل ذخیره شده: {file_size_mb:.2f} MB")

        writer.close()

    cleanup_ddp()

if __name__ == "__main__":
    main()
