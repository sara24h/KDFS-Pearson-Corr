import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

# ==================== تنظیمات اولیه ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ دستگاه استفاده شده: {DEVICE}")

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MODEL_PATH = "/kaggle/working/finetuned_pruned_model.pt"
INPUT_MODEL_PATH = '/kaggle/input/10k_final/pytorch/default/1/10k_final.pt'

DATA_PATHS = {
    "test": [
        "/kaggle/input/wild-deepfake/test/real",
        "/kaggle/input/wild-deepfake/test/fake",
    ],
    "train": [
        "/kaggle/input/wild-deepfake/train/real",
        "/kaggle/input/wild-deepfake/train/fake",
    ],
    "valid": [
        "/kaggle/input/wild-deepfake/valid/real",
        "/kaggle/input/wild-deepfake/valid/fake",
    ]
}

# ==================== تعریف Dataset Custom ====================
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        # بارگذاری تصاویر Real
        for img in os.listdir(real_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(real_dir, img))
                self.labels.append(0)  # 0 = Real
        
        # بارگذاری تصاویر Fake
        for img in os.listdir(fake_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(fake_dir, img))
                self.labels.append(1)  # 1 = Fake
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        except Exception as e:
            print(f"خطا در خواندن تصویر {self.images[idx]}: {e}")
            return None, None

# ==================== تحویل Dataset ====================
def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(labels)

# ==================== تحضیر داده ====================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4414, 0.3448, 0.3159],
                        std=[0.1854, 0.1623, 0.1562])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4414, 0.3448, 0.3159],
                        std=[0.1854, 0.1623, 0.1562])
])

print("\n📊 بارگذاری دیتاست...")

train_dataset = DeepfakeDataset(
    DATA_PATHS["train"][0], 
    DATA_PATHS["train"][1], 
    transform_train
)
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

valid_dataset = DeepfakeDataset(
    DATA_PATHS["valid"][0], 
    DATA_PATHS["valid"][1], 
    transform_test
)
valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

test_dataset = DeepfakeDataset(
    DATA_PATHS["test"][0], 
    DATA_PATHS["test"][1], 
    transform_test
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

print(f"✅ تعداد نمونه‌های آموزش: {len(train_dataset)}")
print(f"✅ تعداد نمونه‌های اعتبارسنجی: {len(valid_dataset)}")
print(f"✅ تعداد نمونه‌های تست: {len(test_dataset)}")

# ==================== لود مدل ====================
print("\n🔧 لود مدل هرس‌شده...")

try:
    # لود چک‌پوینت کامل ورودی
    checkpoint_loaded = torch.load(INPUT_MODEL_PATH, map_location=DEVICE)
    
    # استخراج اطلاعات کلیدی
    model_state_dict = checkpoint_loaded['model_state_dict']
    masks = checkpoint_loaded['masks']
    
    # تبدیل ماسک‌ها به requires_grad=False
    # این کار جلوی مشکل view را می‌گیرد
    if isinstance(masks, dict):
        masks = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v 
                 for k, v in masks.items()}
    elif isinstance(masks, list):
        masks = [m.detach().clone() if isinstance(m, torch.Tensor) else m 
                 for m in masks]
    
    # ساخت مدل هرس‌شده با استفاده از ماسک‌ها
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    # لود وزن‌های هرس‌شده
    model.load_state_dict(model_state_dict, strict=False)
    
    model = model.to(DEVICE)
    
    # اطمینان از اینکه مدل در حالت train است
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("✅ مدل هرس‌شده با موفقیت بازسازی و لود شد!")
    print(f"📊 تعداد کل پارامترها: {total_params:,}")
    print(f"📊 تعداد پارامترهای قابل آموزش: {trainable_params:,}")
    
except Exception as e:
    print(f"❌ خطا در لود مدل: {e}")
    import traceback
    traceback.print_exc()
    raise

# ==================== تنظیمات آموزش ====================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# ==================== تابع آموزش ====================
def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"آموزش Epoch {epoch+1}/{EPOCHS}", 
                       unit='batch', colour='green')
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        if images is None:
            continue
        
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            # پاک کردن gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            outputs = model(images)
            
            # محاسبه loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # بروزرسانی وزن‌ها
            optimizer.step()
            
            # محاسبه آمار
            total_loss += loss.item()
            with torch.no_grad():
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            accuracy = (correct / total) * 100 if total > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })
            
        except RuntimeError as e:
            print(f"\n❌ خطا در batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_accuracy = (correct / total) * 100 if total > 0 else 0
    
    return avg_loss, avg_accuracy

# ==================== تابع اعتبارسنجی ====================
def validate(model, val_loader, criterion, device, phase="Validation"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"{phase}", 
                           unit='batch', colour='blue')
        
        for images, labels in progress_bar:
            if images is None:
                continue
            
            try:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except RuntimeError as e:
                print(f"\n❌ خطا در validation: {e}")
                continue
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    if len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    else:
        accuracy = precision = recall = f1 = 0
    
    return avg_loss, accuracy, precision, recall, f1

# ==================== حلقه آموزش ====================
print("\n🚀 شروع فاین‌تیون مدل...")
print("=" * 80)

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': []
}

best_accuracy = 0
best_epoch = 0

for epoch in range(EPOCHS):
    print(f"\n📅 Epoch {epoch+1}/{EPOCHS}")
    print("-" * 80)
    
    # آموزش
    train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                        optimizer, epoch, DEVICE)
    
    # اعتبارسنجی
    val_loss, val_acc, val_prec, val_recall, val_f1 = validate(
        model, valid_loader, criterion, DEVICE, "اعتبارسنجی"
    )
    
    # ذخیره‌ی نتایج
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_precision'].append(val_prec)
    history['val_recall'].append(val_recall)
    history['val_f1'].append(val_f1)
    
    print(f"\n✅ آموزش - Loss: {train_loss:.4f} | دقت: {train_acc:.2f}%")
    print(f"✅ اعتبارسنجی - Loss: {val_loss:.4f} | دقت: {val_acc:.2f}%")
    print(f"   Precision: {val_prec:.2f}% | Recall: {val_recall:.2f}% | F1: {val_f1:.2f}%")
    
    scheduler.step()
    
    # ذخیره بهترین مدل
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_epoch = epoch
        checkpoint_to_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'masks': masks,
            'accuracy': val_acc,
            'model_architecture': 'ResNet_50_pruned_hardfakevsreal',
            'hyperparameters': {
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE
            }
        }
        torch.save(checkpoint_to_save, MODEL_PATH)
        print(f"💾 بهترین مدل ذخیره شد! (دقت: {best_accuracy:.2f}%)")

print(f"\n🏆 بهترین دقت: {best_accuracy:.2f}% در epoch {best_epoch+1}")

# ==================== تست نهایی ====================
print("\n" + "=" * 80)
print("🧪 شروع تست نهایی...")
print("=" * 80)

# لود بهترین مدل
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc, test_prec, test_recall, test_f1 = validate(
    model, test_loader, criterion, DEVICE, "تست نهایی"
)

print("\n" + "=" * 80)
print("📊 نتایج نهایی:")
print("=" * 80)
print(f"✅ دقت کلی: {test_acc:.2f}%")
print(f"✅ Precision: {test_prec:.2f}%")
print(f"✅ Recall: {test_recall:.2f}%")
print(f"✅ F1-Score: {test_f1:.2f}%")
print(f"✅ Loss: {test_loss:.4f}")

# ==================== ذخیره‌ی گزارش ====================
report = {
    'timestamp': datetime.now().isoformat(),
    'model_path': MODEL_PATH,
    'best_epoch': best_epoch + 1,
    'best_epoch_accuracy': best_accuracy,
    'test_results': {
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_recall,
        'f1_score': test_f1,
        'loss': test_loss
    },
    'hyperparameters': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'optimizer': 'Adam',
        'scheduler': 'StepLR (step_size=2, gamma=0.1)'
    },
    'training_history': history,
    'dataset_sizes': {
        'train': len(train_dataset),
        'validation': len(valid_dataset),
        'test': len(test_dataset)
    }
}

report_path = '/kaggle/working/training_report.json'
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n💾 گزارش نهایی ذخیره شد!")
print(f"📁 مدل: {MODEL_PATH}")
print(f"📁 گزارش: {report_path}")
print("\n✅ فاین‌تیونینگ با موفقیت به پایان رسید!")
