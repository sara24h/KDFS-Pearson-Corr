import json
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import argparse
from model.pruned_model.Resnet_final import ResNet_50_pruned_hardfakevsreal
import pandas as pd
from pathlib import Path

# ---------- Dataset جدید بر اساس CSV ----------
class CSVImageDataset(Dataset):
    def __init__(self, csv_file, real_dir, fake_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        self.transform = transform
        self.df['label'] = self.df['label'].apply(lambda x: 1 if str(x).strip() in ['1', '1.0'] else 0)
        print(f"📊 Loaded {len(self.df)} samples from {csv_file}")
        print(f"   Real (label=1): {self.df['label'].sum()}")
        print(f"   Fake (label=0): {len(self.df) - self.df['label'].sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label = int(row['label'])
        img_path = (self.real_dir / filename) if label == 1 else (self.fake_dir / filename)
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(float(label), dtype=torch.float32)
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            return torch.zeros(3, 256, 256), torch.tensor(float(label), dtype=torch.float32)

# ---------- Transform ها ----------
train_transform = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4868, 0.3972, 0.3624], std=[0.2296, 0.2066, 0.2009]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4868, 0.3972, 0.3624], std=[0.2296, 0.2066, 0.2009])
])

# ---------- DataLoaders بدون DDP ----------
def create_dataloaders(batch_size=256, num_workers=4):
    REAL_DIR = "/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/my_real_vs_ai_dataset/my_real_vs_ai_dataset/real"
    FAKE_DIR = "/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/my_real_vs_ai_dataset/my_real_vs_ai_dataset/ai_images"

    train_dataset = CSVImageDataset(
        csv_file="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/train.csv",
        real_dir=REAL_DIR,
        fake_dir=FAKE_DIR,
        transform=train_transform
    )

    val_dataset = CSVImageDataset(
        csv_file="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/valid.csv",
        real_dir=REAL_DIR,
        fake_dir=FAKE_DIR,
        transform=val_transform
    )

    test_dataset = CSVImageDataset(
        csv_file="/kaggle/working/Test/test_labels.csv",  # ⚠️ نیاز به CSV برای تست!
        real_dir="/kaggle/working/Test/real",
        fake_dir="/kaggle/working/Test/fake",
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# ---------- Train و Validate بدون DDP ----------
def train_epoch(model, loader, criterion, optimizer, device, scaler, writer, epoch, accum_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training")

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

        pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}', 'acc': f'{100. * correct / total:.2f}%'})

    avg_loss = running_loss / len(loader)
    avg_acc = 100. * correct / total

    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/acc", avg_acc, epoch)

    return avg_loss, avg_acc

@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Validation"):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(loader)
    avg_acc = 100. * correct / total

    if writer:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/acc", avg_acc, epoch)

    return avg_loss, avg_acc

# ---------- Main بدون DDP ----------
def main(args):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    BASE_LR = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    ACCUM_STEPS = args.accum_steps

    writer = SummaryWriter('/kaggle/working/runs')

    print("=" * 70)
    print("🚀 Fine-tuning on 200k Dataset (Single GPU)")
    print(f"   Batch Size: {BATCH_SIZE}, Accum Steps: {ACCUM_STEPS}")
    print(f"   Effective Batch: {BATCH_SIZE * ACCUM_STEPS}")
    print("=" * 70)

    # لود مدل
    input_model_path = '/kaggle/working/330k_final.pt'
    checkpoint = torch.load(input_model_path, map_location=DEVICE)
    masks_detached = [m.detach().clone() if m is not None else None for m in checkpoint['masks']]

    model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    # فریز و آزادسازی لایه‌ها
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters(): param.requires_grad = True
    for param in model.layer4.parameters(): param.requires_grad = True
    for param in model.fc.parameters():     param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model loaded. Trainable: {trainable_params:,} / Total: {total_params:,}")

    # دیتالودرها
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': BASE_LR * 0.3, 'weight_decay': WEIGHT_DECAY * 1.5},
        {'params': model.layer4.parameters(), 'lr': BASE_LR * 0.6, 'weight_decay': WEIGHT_DECAY * 1.5},
        {'params': model.fc.parameters(),     'lr': BASE_LR * 1.0, 'weight_decay': WEIGHT_DECAY * 2.5}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler(enabled=True)

    best_val_acc = 0.0
    best_model_path = '/kaggle/working/best_pruned_finetuned_200k.pt'

    for epoch in range(NUM_EPOCHS):
        print(f"\n📍 Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 70)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler, writer, epoch, ACCUM_STEPS)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, writer, epoch)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved with Val Acc: {val_acc:.2f}%")

        scheduler.step()

    # تست نهایی
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, writer, NUM_EPOCHS)
    print("\n" + "=" * 70)
    print("🧪 Final Test Results")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # ذخیره مدل استنتاج
    model_inference = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])
    model_inference.load_state_dict(model.state_dict())
    model_inference = model_inference.cpu().eval()

    torch.save({
        'model_state_dict': model_inference.state_dict(),
        'total_params': sum(p.numel() for p in model_inference.parameters()),
        'masks': checkpoint['masks'],
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
    }, '/kaggle/working/final_pruned_finetuned_200k_inference.pt')

    print("✅ Inference-ready model saved!")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--accum_steps', type=int, default=1)
    args = parser.parse_args()
    main(args)
