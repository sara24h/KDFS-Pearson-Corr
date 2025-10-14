import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ⚠️ اطمینان حاصل کنید که این کلاس مدل در محیط شما تعریف شده و قابل Import است
# این مدل از 'masks' برای ساختار هرس‌شده استفاده می‌کند
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal 


# ==================== ۱. دیتاست سفارشی ====================
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        # بارگذاری تصاویر Real
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)  # Real = 0
        
        # بارگذاری تصاویر Fake
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1)  # Fake = 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== ۲. تنظیمات عمومی و داده‌ها ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"استفاده از دستگاه: {device}")

# Transformations (با استفاده از آمار دیتاست شما)
mean = [0.4414, 0.3448, 0.3159]
std = [0.1854, 0.1623, 0.1562]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# مسیرهای دیتاست
BASE_DIR = "/kaggle/input/wild-deepfake"

train_dataset = DeepfakeDataset(
    real_dir=os.path.join(BASE_DIR, "train/real"), fake_dir=os.path.join(BASE_DIR, "train/fake"), transform=train_transform
)
valid_dataset = DeepfakeDataset(
    real_dir=os.path.join(BASE_DIR, "valid/real"), fake_dir=os.path.join(BASE_DIR, "valid/fake"), transform=test_transform
)
test_dataset = DeepfakeDataset(
    real_dir=os.path.join(BASE_DIR, "test/real"), fake_dir=os.path.join(BASE_DIR, "test/fake"), transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# ==================== ۳. بارگذاری و بازسازی مدل هرس‌شده (نسخه نهایی اصلاح‌شده) ====================
print("\n🔄 در حال بازسازی و لود مدل هرس‌شده...")

# ⚠️ مسیر فایل چک‌پوینت شما
CHECKPOINT_PATH = '/kaggle/input/10k_final/pytorch/default/1/10k_final.pt' 

try:
    # 1. لود چک‌پوینت (دیکشنری کامل)
    checkpoint_loaded = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # 2. استخراج وزن‌ها و ماسک‌ها با بررسی کلیدهای محتمل
    
    # اولویت اول: کلید صحیح که در آخرین خروجی شما (model_state_dict) مشاهده شد
    if 'model_state_dict' in checkpoint_loaded:
        model_state_dict = checkpoint_loaded['model_state_dict']
    # اولویت دوم: کلید student (اگر فایل چک‌پوینت از مرحله KD/Pruning باشد)
    elif 'student' in checkpoint_loaded:
        model_state_dict = checkpoint_loaded['student']
    else:
        raise KeyError("هیچ یک از کلیدهای 'model_state_dict' یا 'student' برای وزن‌های مدل یافت نشد.")

    # ماسک‌ها
    masks = checkpoint_loaded.get('masks')
    if masks is None:
        # اگر ماسک‌ها پیدا نشدند، مدل را نمی‌توان به درستی بازسازی کرد
        raise KeyError("کلید 'masks' در چک‌پوینت برای بازسازی مدل هرس‌شده یافت نشد.")

    # 3. ساخت مدل هرس‌شده با استفاده از ماسک‌ها
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    # 4. لود وزن‌ها
    model.load_state_dict(model_state_dict)
    
    # 5. انتقال مدل به دستگاه (GPU/CPU)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ مدل هرس‌شده با موفقیت لود و بازسازی شد! تعداد پارامترها: {total_params:,}")

except Exception as e:
    print(f"❌ خطا در لود و بازسازی مدل هرس‌شده: {e}")
    print(f"⚠️ جزئیات خطا: {type(e).__name__}: {e}")
    # خروج از برنامه در صورت عدم موفقیت در لود مدل
    exit() 

# ==================== ۴. تنظیمات Fine-tuning ====================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

num_epochs = 20
best_val_loss = float('inf')

# ==================== ۵. توابع آموزش و ارزیابی ====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc='Training', leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc='Validation', leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total

# ==================== ۶. اجرای Fine-tuning ====================
print("\n🚀 شروع Fine-tuning مدل هرس‌شده...")
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
    print("-" * 50)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, valid_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    scheduler.step(val_loss)
    
    # ذخیره بهترین مدل (فقط state_dict)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # ذخیره وزن‌ها و ماسک‌ها برای بارگذاری آسان در آینده
        torch.save({
            'model_state_dict': model.state_dict(),
            'masks': masks 
        }, 'best_finetuned_model_weights.pt')
        print("✅ مدل بهبود یافت و ذخیره شد!")

# ==================== ۷. تست نهایی ====================
print("\n🧪 شروع تست نهایی...")

# بازسازی مدل برای تست با بهترین وزن‌ها
try:
    model_test = ResNet_50_pruned_hardfakevsreal(masks=masks)
    best_weights = torch.load('best_finetuned_model_weights.pt', map_location=device)
    model_test.load_state_dict(best_weights['model_state_dict'])
    model_test = model_test.to(device)
    model_test.eval()
except Exception as e:
    print(f"❌ خطا در لود مدل نهایی برای تست: {e}. استفاده از آخرین مدل آموزش‌دیده.")
    model_test = model # استفاده از مدل فعلی در حافظه
    model_test.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing'):
        images = images.to(device)
        outputs = model_test(images)
        predicted = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
        
        all_preds.extend(predicted.flatten())
        all_labels.extend(labels.numpy())

# محاسبه متریک‌ها
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "="*60)
print("📈 نتایج تست نهایی:")
print("="*60)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print("\nConfusion Matrix:")
print(cm)

# رسم نمودارها
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
axes[0].plot(train_losses, label='Train Loss', marker='o')
axes[0].plot(val_losses, label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy curve
axes[1].plot(train_accs, label='Train Acc', marker='o')
axes[1].plot(val_accs, label='Val Acc', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')
axes[2].set_title('Confusion Matrix')
axes[2].set_xticklabels(['Real', 'Fake'])
axes[2].set_yticklabels(['Real', 'Fake'])

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ فرآیند Fine-tuning و تست با موفقیت به پایان رسید!")
print(f"📁 مدل نهایی (وزن‌ها) در 'best_finetuned_model_weights.pt' ذخیره شد")
print(f"📊 نمودارها در 'training_results.png' ذخیره شدند")
