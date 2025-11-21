import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import sys
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal


class WildDeepFakeDataset(Dataset):
    """دیتاست برای لود کردن تصاویر fake و real"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # خواندن فایل‌ها از پوشه‌های fake و real
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            label
                        ))
        
        print(f"✅ تعداد نمونه‌های لود شده از {root_dir}: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(phase, mean, std):
    """تعریف transformations برای train و validation/test"""
    
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def load_pruned_model(checkpoint_path, device):
    """لود کردن مدل هرس‌شده از checkpoint"""
    
    print(f"📥 در حال لود کردن مدل از {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # استخراج masks از checkpoint
    masks = checkpoint['masks']
    
    # ساخت مدل با masks
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    # لود کردن weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ مدل با موفقیت لود شد!")
    print(f"📊 تعداد پارامترها: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):

    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"🚀 Epoch {epoch} [Train]")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs, _ = model(images)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # محاسبه metrics
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, phase='Valid'):
    """ارزیابی مدل روی validation یا test set"""
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"🔍 {phase}")
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs, _ = model(images)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # محاسبه metrics اضافی
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(all_labels, all_probs)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def fine_tune(
    model_path,
    data_dir,
    output_dir,
    mean=[0.5207,0.4258,0.3806],
    std=[0.2490,0.2239,0.2212],
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=10
):

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_pruned_model(model_path, device)

    train_transform = get_transforms('train', mean, std)
    val_transform = get_transforms('val', mean, std)
    
    # لود دیتاست‌ها
    print("\n📁 در حال لود کردن دیتاست‌ها...")
    train_dataset = WildDeepFakeDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    valid_dataset = WildDeepFakeDataset(
        os.path.join(data_dir, 'valid'),
        transform=val_transform
    )
    test_dataset = WildDeepFakeDataset(
        os.path.join(data_dir, 'test'),
        transform=val_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # تعریف loss و optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # متغیرهای early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    print("\n" + "="*70)
    print("🚀 شروع Fine-tuning")
    print("="*70)
    
    # حلقه آموزش
    for epoch in range(1, num_epochs + 1):
        print(f"\n📍 Epoch {epoch}/{num_epochs}")
        print("-" * 70)
        
        # آموزش
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # اعتبارسنجی
        val_metrics = validate(model, valid_loader, criterion, device, 'Valid')
        
        # نمایش نتایج
        print(f"\n📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"📊 Valid Loss: {val_metrics['loss']:.4f} | Valid Acc: {val_metrics['accuracy']:.4f}")
        print(f"📊 Valid F1: {val_metrics['f1']:.4f} | Valid AUC: {val_metrics['auc']:.4f}")
        
        # بروزرسانی learning rate
        scheduler.step(val_metrics['loss'])
        
        # ذخیره بهترین مدل
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            epochs_no_improve = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'masks': model.masks if hasattr(model, 'masks') else None,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'val_metrics': val_metrics
            }
            
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            print(f"💾 مدل بهتری ذخیره شد! (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⏳ {epochs_no_improve}/{patience} epochs بدون بهبود")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n⛔ Early stopping! بهترین Val Loss: {best_val_loss:.4f}")
            break
    
    # ارزیابی نهایی روی test set
    print("\n" + "="*70)
    print("🧪 ارزیابی نهایی روی Test Set")
    print("="*70)
    
    # لود بهترین مدل
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device, 'Test')
    
    print(f"\n🎯 نتایج نهایی:")
    print(f"   Test Loss:      {test_metrics['loss']:.4f}")
    print(f"   Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Test Precision: {test_metrics['precision']:.4f}")
    print(f"   Test Recall:    {test_metrics['recall']:.4f}")
    print(f"   Test F1-Score:  {test_metrics['f1']:.4f}")
    print(f"   Test AUC:       {test_metrics['auc']:.4f}")
    
    # ذخیره نتایج
    results = {
        'best_epoch': best_checkpoint['epoch'],
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics
    }
    
    torch.save(results, os.path.join(output_dir, 'training_results.pt'))
    
    print(f"\n✅ Fine-tuning تمام شد!")
    print(f"📁 فایل‌های خروجی در {output_dir} ذخیره شدند")
    
    return model, results


if __name__ == "__main__":
    # تنظیمات
    MODEL_PATH = '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt'  # مسیر مدل هرس‌شده
    DATA_DIR = '/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k'    # مسیر دیتاست
    OUTPUT_DIR = '/kaggle/working/140k_finetuned_pruned_model' # مسیر خروجی
    

    MEAN = [0.5207,0.4258,0.3806]  # ImageNet defaults
    STD = [0.2490,0.2239,0.2212]   # ImageNet defaults

    model, results = fine_tune(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        mean=MEAN,
        std=STD,
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-4,
        patience=10
    )
