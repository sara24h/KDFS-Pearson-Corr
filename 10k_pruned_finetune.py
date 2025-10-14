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
print(f"دستگاه استفاده شده: {DEVICE}")

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MODEL_PATH = "/kaggle/working/finetuned_pruned_model.pt"
INPUT_MODEL_PATH = '/kaggle/input/10k_pruned_model_resnet50/pytorch/default/1/resnet50_pruned_model_learnable_masks.pt'

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
        except:
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
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
    num_workers=2
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
    num_workers=2
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
    num_workers=2
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
    
    # ساخت مدل هرس‌شده با استفاده از ماسک‌ها
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    # لود وزن‌های هرس‌شده
    model.load_state_dict(model_state_dict)
    
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print("✅ مدل هرس‌شده با موفقیت بازسازی و لود شد!")
    print(f"تعداد پارامترها: {total_params:,}")
    
except Exception as e:
    print(f"❌ خطا در لود مدل: {e}")
    raise

# ==================== تنظیمات آموزش ====================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'accuracy': f'{accuracy:.2f}%'
        })
    
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
                
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) * 100 if len(all_labels) > 0 else 0
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    
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
    
    print(f"✅ آموزش - Loss: {train_loss:.4f} | دقت: {train_acc:.2f}%")
    print(f"✅ اعتبارسنجی - Loss: {val_loss:.4f} | دقت: {val_acc:.2f}%")
    print(f"   Precision: {val_prec:.2f}% | Recall: {val_recall:.2f}% | F1: {val_f1:.2f}%")
    
    scheduler.step()
    
    # ذخیره بهترین مدل
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        checkpoint_to_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'masks': masks,
            'accuracy': val_acc,
            'model_architecture': 'ResNet_50_pruned_hardfakevsreal'
        }
        torch.save(checkpoint_to_save, MODEL_PATH)
        print(f"💾 بهترین مدل ذخیره شد! (دقت: {best_accuracy:.2f}%)")

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
        'learning_rate': LEARNING_RATE
    },
    'training_history': history
}

with open('/kaggle/working/training_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n💾 گزارش نهایی ذخیره شد!")
print(f"📁 مدل: {MODEL_PATH}")
print(f"📁 گزارش: /kaggle/working/training_report.json")
