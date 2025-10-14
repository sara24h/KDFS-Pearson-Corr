import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

# فرض می‌کنیم کلاس مدل شما در این مسیر قابل دسترسی است
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def main():
    # --- 1. تنظیمات اولیه ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("======================================================================")
    print("🚀 شروع Fine-tuning مدل Pruned ResNet50")
    print("======================================================================")

    # مسیر مدل ذخیره شده (خروجی کد قبلی)
    model_save_path = '/kaggle/working/resnet50_pruned_reloaded_saved.pt'

    # مسیرهای داده
    data_dir = '/kaggle/input/wild-deepfake'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # --- 2. تعریف تبدیلات (Transforms) ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 3. لود داده‌ها ---
    print("📊 آماده‌سازی DataLoaders...")
    try:
        image_datasets = {
            'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
            'val': datasets.ImageFolder(root=val_dir, transform=data_transforms['val']),
            'test': datasets.ImageFolder(root=test_dir, transform=data_transforms['test']),
        }
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=2, pin_memory=True),
            'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=2, pin_memory=True),
            'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=2, pin_memory=True),
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        class_names = image_datasets['train'].classes
        print(f"📊 Dataset loaded: {dataset_sizes['train']} images ({dataset_sizes['train']//2} real, {dataset_sizes['train']//2} fake)")
        print(f"📊 Dataset loaded: {dataset_sizes['val']} images ({dataset_sizes['val']//2} real, {dataset_sizes['val']//2} fake)")
        print(f"📊 Dataset loaded: {dataset_sizes['test']} images ({dataset_sizes['test']//2} real, {dataset_sizes['test']//2} fake)")
    except FileNotFoundError as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"❌ Check if paths exist: {train_dir}, {val_dir}, {test_dir}")
        return

    # --- 4. لود مدل هرس‌شده ---
    print("📦 لود مدل Pruned...")
    try:
        checkpoint = torch.load(model_save_path, map_location=device)
        masks = checkpoint['masks']

        model = ResNet_50_pruned_hardfakevsreal(masks=masks)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("✅ مدل لود شد")
        print(f"   - تعداد کل پارامترها: {total_params:,}")
        print(f"   - تعداد پارامترهای قابل آموزش: {trainable_params:,}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- 5. تعریف تابع خطا و بهینه‌ساز ---
    criterion = nn.CrossEntropyLoss()
    # نرخ یادگیری کم برای Fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4) # weight_decay ممکن است کمک کند
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 6. حلقه آموزش (Fine-tuning) ---
    num_epochs = 20 # تعداد اپوک‌های مورد نظر
    best_val_acc = 0.0
    best_model_wts = model.state_dict().copy()

    print("======================================================================")
    print("🎓 شروع آموزش")
    print("======================================================================")

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n📍 Epoch {epoch+1}/{num_epochs}")
        print(f"   Learning Rate: {current_lr:.6f}")
        print("-" * 70)

        # آموزش
        train_loss, train_acc = train_epoch(model, dataloaders['train'], criterion, optimizer, device, scheduler, epoch, num_epochs)
        print(f"   Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        # ارزیابی
        val_loss, val_acc, val_f1, _, _ = val_epoch(model, dataloaders['val'], criterion, device, epoch, num_epochs)
        print(f"   Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # ذخیره بهترین مدل
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            print("   -> Best model updated!")

    print(f"\n🎉 Best val Acc: {best_val_acc:.4f}")

    # --- 7. ذخیره مدل Fine-tune شده ---
    model.load_state_dict(best_model_wts)
    fine_tuned_save_path = '/kaggle/working/resnet50_pruned_finetuned_wilddeepfake.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'masks': checkpoint.get('masks'), # از چک‌پوینت اصلی استفاده می‌شود
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'epoch': epoch+1,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal_finetuned'
    }, fine_tuned_save_path)

    print(f"✅ Fine-tuned model saved to {fine_tuned_save_path}")

    # --- 8. ارزیابی نهایی روی داده‌های تست ---
    print("\n======================================================================")
    print("🧪 ارزیابی نهایی روی داده‌های تست")
    print("======================================================================")
    model.load_state_dict(best_model_wts) # اطمینان از استفاده از بهترین مدل
    test_loss, test_acc, test_f1, test_labels, test_preds = val_epoch(model, dataloaders['test'], criterion, device, 'Final_Test', 0)

    print("\nFinal Evaluation on Test Set:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    print("\n🎉 اتمام فرآیند Fine-tuning و ارزیابی.")


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    progress_bar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{num_epochs}', leave=False)

    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward - بدون autocast
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # بهینه‌سازی
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{running_corrects / ((progress_bar.n + 1) * progress_bar.batch_size):.4f}'
        })

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def val_epoch(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f'Val/Test Epoch {epoch+1}/{num_epochs}' if isinstance(epoch, int) else 'Testing', leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward - بدون autocast
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{running_corrects / ((progress_bar.n + 1) * progress_bar.batch_size):.4f}'
            })

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, val_f1, all_labels, all_preds


if __name__ == "__main__":
    main()
