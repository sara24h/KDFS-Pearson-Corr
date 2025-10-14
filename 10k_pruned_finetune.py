import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal # یا نام فایل شما

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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# ============================================================
# 4. تابع آموزش
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device) # بدون unsqueeze

        # فقط خروجی اول مدل را بگیر
        outputs, _ = model(inputs) # <--- تغییر اینجا

        # **راه‌حل کلیدی: استفاده از autocast برای mixed precision**
        with torch.cuda.amp.autocast(enabled=True):
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        # Backward با scaler
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}', 
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================
# 5. تابع اعتبارسنجی
# ============================================================
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Validation"):
        inputs, labels = inputs.to(device), labels.to(device) # بدون unsqueeze

        # فقط خروجی اول مدل را بگیر
        outputs, _ = model(inputs) # <--- تغییر اینجا
        
        with torch.cuda.amp.autocast(enabled=True):
            loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================
# 6. اصلی برنامه Fine-tuning
# ============================================================
def main():
    # تنظیمات
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16  # کاهش batch size برای جلوگیری از مشکل memory
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001  # کاهش LR برای fine-tuning روی مدل pruned
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 2  # برای شبیه‌سازی batch size بزرگتر
    
    print("="*70)
    print("🚀 شروع Fine-tuning مدل Pruned ResNet50")
    print("="*70)
    
    # لود مدل
    print("\n📦 لود مدل Pruned...")
    input_model_path = '/kaggle/input/m/saraaskari/10k_final/pytorch/default/1/10k_final.pt'
    checkpoint = torch.load(input_model_path, map_location=DEVICE)
    
    # ساخت مدل با detach کردن masks
    masks_detached = [m.detach().clone() if m is not None else None for m in checkpoint['masks']]
    
    model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ مدل لود شد")
    print(f"   - تعداد کل پارامترها: {total_params:,}")
    print(f"   - تعداد پارامترهای قابل آموزش: {trainable_params:,}")
    
    # آماده‌سازی داده‌ها
    print("\n📊 آماده‌سازی DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE, 
        num_workers=2  # کاهش برای stability
    )
    
    # تعریف Loss و Optimizer
    # چون num_classes=1 است، BCEWithLogitsLoss مناسب است
    criterion = nn.BCEWithLogitsLoss()
    
    # فقط لایه‌های خاص را train کنیم (optional)
    # برای شروع، همه لایه‌ها را train می‌کنیم
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # آموزش
    print("\n" + "="*70)
    print("🎓 شروع آموزش")
    print("="*70)
    
    best_val_acc = 0.0
    best_model_path = '/kaggle/working/best_pruned_finetuned.pt'
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📍 Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 70)
        
        # آموزش
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, ACCUMULATION_STEPS
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # اعتبارسنجی
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # به‌روزرسانی Learning Rate
        scheduler.step()
        
        # ذخیره بهترین مدل
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'masks': checkpoint['masks'], # ذخیره ماسک‌ها
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'total_params': total_params
            }, best_model_path)
            print(f"💾 بهترین مدل ذخیره شد (Val Acc: {val_acc:.2f}%)")
    
    # تست نهایی
    print("\n" + "="*70)
    print("🧪 تست نهایی با بهترین مدل")
    print("="*70)
    
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # ذخیره مدل نهایی
    final_model_path = '/kaggle/working/final_pruned_finetuned.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'masks': checkpoint['masks'], # ذخیره ماسک‌ها
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

if __name__ == "__main__":
    main()
