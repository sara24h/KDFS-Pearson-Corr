import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from sklearn.metrics import accuracy_score, classification_report, f1_score

# --- 1. تنظیمات اولیه ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# مسیر مدل ذخیره شده (خروجی کد قبلی)
model_save_path = '/kaggle/working/resnet50_pruned_reloaded_saved.pt'

# مسیرهای داده
data_dir = '/kaggle/input/wild-deepfake'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test') # اضافه شد

# --- 2. تعریف تبدیلات (Transforms) ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.4414, 0.3448, 0.3159], [0.1854, 0.1623, 0.1562])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4414, 0.3448, 0.3159], [0.1854, 0.1623, 0.1562])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4414, 0.3448, 0.3159], [0.1854, 0.1623, 0.1562])
    ]),
}

# --- 3. لود داده‌ها ---
print("Loading datasets...")
try:
    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=val_dir, transform=data_transforms['val']),
        'test': datasets.ImageFolder(root=test_dir, transform=data_transforms['test']), # اضافه شد
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=2),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=2), # اضافه شد
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']} # 'test' اضافه شد
    class_names = image_datasets['train'].classes
    print(f"Classes found: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    print(f"Check if paths exist: {train_dir}, {val_dir}, {test_dir}")
    exit()

# --- 4. لود مدل هرس‌شده ---
print("Loading the pruned model...")
checkpoint = torch.load(model_save_path, map_location=device)
masks = checkpoint['masks']

model = ResNet_50_pruned_hardfakevsreal(masks=masks)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.train()

# --- 5. تعریف تابع خطا و بهینه‌ساز ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- 6. حلقه آموزش (Fine-tuning) با نوار پیشرفت ---
num_epochs = 10
best_val_acc = 0.0
best_model_wts = model.state_dict().copy()

print("Starting Fine-tuning...")
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        # استفاده از tqdm برای نوار پیشرفت
        progress_bar = tqdm(dataloaders[phase], desc=f'{phase} ', leave=False)
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

            # به‌روزرسانی نوار پیشرفت
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{running_corrects / (len(progress_bar) * progress_bar.batch_size):.4f}'
            })

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        if phase == 'train':
            scheduler.step()

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'val' and epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_model_wts = model.state_dict().copy()
            print("  -> Best model updated!")

    print()

print(f'Best val Acc: {best_val_acc:4f}')

# --- 7. ذخیره مدل Fine-tune شده ---
model.load_state_dict(best_model_wts)
fine_tuned_save_path = '/kaggle/working/resnet50_pruned_finetuned_wilddeepfake.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'masks': masks,
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
    'epoch': epoch+1,
    'model_architecture': 'ResNet_50_pruned_hardfakevsreal_finetuned'
}, fine_tuned_save_path)

print(f"Fine-tuned model saved to {fine_tuned_save_path}")

# --- 8. ارزیابی نهایی روی داده‌های تست ---
print("\nStarting evaluation on Test Set...")
model.eval()
all_test_preds = []
all_test_labels = []

# استفاده از tqdm برای تست نیز
progress_bar_test = tqdm(dataloaders['test'], desc='Testing ', leave=False)
with torch.no_grad():
    for inputs, labels in progress_bar_test:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_test_labels, all_test_preds)
test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')

print("\nFinal Evaluation on Test Set:")
print(f"Accuracy: {test_acc:.4f}")
print(f"F1-Score: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_test_labels, all_test_preds, target_names=class_names))
