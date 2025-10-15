import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
        
        # لود تصاویر Real (label = 0)
        if os.path.exists(real_path):
            real_files = [f for f in os.listdir(real_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in real_files:
                self.images.append(os.path.join(real_path, fname))
                self.labels.append(0)
        
        # لود تصاویر Fake (label = 1)
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
            # برای BCEWithLogitsLoss با num_classes=1، لیبل باید float باشد
            return img, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float32)

# ============================================================
# 2. تعریف Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 3. آماده‌سازی DataLoaders
# ============================================================
def create_dataloaders(batch_size=32, num_workers=4):
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
    
    # ایجاد DistributedSampler برای DDP
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler

# ============================================================
# 4. تابع آموزش
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1, rank=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    # فقط یک پیشرفت نشانگر در یک rank نمایش داده شود
    pbar = tqdm(loader, desc="Training", disable=rank != 0)
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)

        outputs, _ = model(inputs)

        with torch.cuda.amp.autocast(enabled=True):
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}', 
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    # میانگین‌گیری بین تمام rank ها
    avg_loss = torch.tensor(running_loss / len(loader)).to(device)
    avg_acc = torch.tensor(100. * correct / total).to(device)
    
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
    
    avg_loss = avg_loss.item() / dist.get_world_size()
    avg_acc = avg_acc.item() / dist.get_world_size()
    
    return avg_loss, avg_acc

@torch.no_grad()
def validate(model, loader, criterion, device, rank=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Validation", disable=rank != 0):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)

        outputs, _ = model(inputs)
        
        with torch.cuda.amp.autocast(enabled=True):
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
    
    return avg_loss, avg_acc

# ============================================================
# 5. تابع setup DDP
# ============================================================
def setup_ddp():
    # در Kaggle یا محیط‌های دیگر، این متغیرها ممکن است از قبل تنظیم شده باشند
    # اما می‌توانید به صورت دستی هم تنظیم کنید
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

# ============================================================
# 6. اصلی برنامه Fine-tuning
# ============================================================
def main():
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    
    # تنظیمات
    DEVICE = torch.device(f"cuda:{local_rank}")
    BATCH_SIZE_PER_GPU = 16  # تعداد نمونه در هر گرافیک
    BATCH_SIZE = BATCH_SIZE_PER_GPU * world_size  # کل نمونه در هر استپ
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 2
    
    if global_rank == 0:
        print("="*70)
        print("🚀 شروع Fine-tuning مدل Pruned ResNet50 با DDP")
        print(f"   تعداد گرافیک: {world_size}")
        print(f"   Batch Size کل: {BATCH_SIZE}")
        print("="*70)
    
    # لود مدل
    if global_rank == 0:
        print("\n📦 لود مدل Pruned...")
    
    input_model_path = '/kaggle/input/m/saraaskari/10k_final/pytorch/default/1/10k_final.pt'
    checkpoint = torch.load(input_model_path, map_location=DEVICE)
    
    masks_detached = [m.detach().clone() if m is not None else None for m in checkpoint['masks']]
    
    model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(DEVICE)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if global_rank == 0:
        print(f"✅ مدل لود شد")
        print(f"   - تعداد کل پارامترها: {total_params:,}")
        print(f"   - تعداد پارامترهای قابل آموزش: {trainable_params:,}")
    
    # آماده‌سازی داده‌ها
    if global_rank == 0:
        print("\n📊 آماده‌سازی DataLoaders...")
    
    train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = create_dataloaders(
        batch_size=BATCH_SIZE_PER_GPU, 
        num_workers=2
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # آموزش
    if global_rank == 0:
        print("\n" + "="*70)
        print("🎓 شروع آموزش")
        print("="*70)
    
    best_val_acc = 0.0
    best_model_path = f'/kaggle/working/best_pruned_finetuned_ddp_rank_{global_rank}.pt'
    
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)  # برای تغییر ترتیب داده‌ها در هر ایپاک
        val_sampler.set_epoch(epoch)
        
        if global_rank == 0:
            print(f"\n📍 Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 70)
        
        # آموزش
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, ACCUMULATION_STEPS, global_rank
        )
        if global_rank == 0:
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # اعتبارسنجی
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, global_rank)
        if global_rank == 0:
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
        
        # ذخیره بهترین مدل فقط در rank 0
        if global_rank == 0:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # model.module چون DDP
                    'masks': checkpoint['masks'],
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'total_params': total_params
                }, best_model_path)
                print(f"💾 بهترین مدل ذخیره شد (Val Acc: {val_acc:.2f}%)")
    
    # تست نهایی
    if global_rank == 0:
        print("\n" + "="*70)
        print("🧪 تست نهایی با بهترین مدل")
        print("="*70)
        
        best_checkpoint = torch.load(best_model_path)
        model.module.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, global_rank)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        final_model_path = '/kaggle/working/final_pruned_finetuned_ddp.pt'
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'masks': checkpoint['masks'],
            'test_acc': test_acc,
            'best_val_acc': best_val_acc,
            'total_params': total_params,
            'model_architecture': 'ResNet_50_pruned_hardfakevsreal'
        }, final_model_path)
        
        print(f"\n✅ مدل نهایی در {final_model_path} ذخیره شد")
        print(f"📊 بهترین دقت Validation: {best_val_acc:.2f}%")
        print(f"📊 دقت Test: {test_acc:.2f}%")
        print("\n" + "="*70)
        print("🎉 Fine-tuning با موفقیت انجام شد!")
        print("="*70)
    
    cleanup_ddp()

if __name__ == "__main__":
    main()
