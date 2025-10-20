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
import pandas as pd


# ---------- Dataset جدید بر اساس CSV ----------
class CSVImageDataset(Dataset):
    def __init__(self, csv_file, real_dir, fake_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        self.transform = transform

        # نرمال‌سازی برچسب: 1 → real, 0 → fake
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

        # تعیین مسیر بر اساس برچسب
        img_path = (self.real_dir / filename) if label == 1 else (self.fake_dir / filename)

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(float(label), dtype=torch.float32)
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            return torch.zeros(3, 256, 256), torch.tensor(float(label), dtype=torch.float32)


# ---------- Transform ها (بدون تغییر) ----------
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


# ---------- DataLoaders جدید ----------
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

    test_dataset = WildDeepfakeDataset(  # فقط برای تست نهایی از پوشه استفاده می‌کنیم
        real_path="/kaggle/working/Test/real",
        fake_path="/kaggle/working/Test/fake",
        transform=val_transform
    )

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


# ---------- باقی کد بدون تغییر (اما با اصلاح کوچک در تست) ----------
# ... (بقیه توابع train_epoch, validate, setup_ddp, cleanup_ddp بدون تغییر)

def main(args):
    SEED = 42
    local_rank = setup_ddp(SEED)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    DEVICE = torch.device(f"cuda:{local_rank}")
    BATCH_SIZE_PER_GPU = args.batch_size
    BATCH_SIZE = BATCH_SIZE_PER_GPU * world_size
    NUM_EPOCHS = args.num_epochs
    BASE_LR = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    ACCUM_STEPS = args.accum_steps

    result_dir = f'/kaggle/working/runs_ddp_rank_{global_rank}'
    writer = SummaryWriter(result_dir) if global_rank == 0 else None

    if global_rank == 0:
        print("=" * 70)
        print("🚀 Fine-tuning on 200k Dataset (CSV-based, Real vs AI)")
        print(f"   Train/Val from CSV | Test from /kaggle/working/Test/")
        print(f"   GPUs: {world_size}, Batch: {BATCH_SIZE}, Accum: {ACCUM_STEPS}")
        print("=" * 70)

    input_model_path = '/kaggle/working/330k_final.pt'
    checkpoint = torch.load(input_model_path, map_location=DEVICE)
    masks_detached = [m.detach().clone() if m is not None else None for m in checkpoint['masks']]

    model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters(): param.requires_grad = True
    for param in model.layer4.parameters(): param.requires_grad = True
    for param in model.fc.parameters():     param.requires_grad = True

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if global_rank == 0:
        print(f"✅ Model loaded. Trainable: {trainable_params:,} / Total: {total_params:,}")

    if global_rank == 0:
        print("\n📊 Preparing DataLoaders...")

    train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = create_dataloaders(
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=2
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW([
        {'params': model.module.layer3.parameters(), 'lr': BASE_LR * 0.3, 'weight_decay': WEIGHT_DECAY * 1.5},
        {'params': model.module.layer4.parameters(), 'lr': BASE_LR * 0.6, 'weight_decay': WEIGHT_DECAY * 1.5},
        {'params': model.module.fc.parameters(),     'lr': BASE_LR * 1.0, 'weight_decay': WEIGHT_DECAY * 2.5}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = GradScaler(enabled=True)

    best_val_acc = 0.0
    best_model_path = '/kaggle/working/best_pruned_finetuned_200k.pt'

    try:
        for epoch in range(NUM_EPOCHS):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            if global_rank == 0:
                print(f"\n📍 Epoch {epoch+1}/{NUM_EPOCHS}")
                print(f"   LR: layer3={optimizer.param_groups[0]['lr']:.7f}, layer4={optimizer.param_groups[1]['lr']:.7f}, fc={optimizer.param_groups[2]['lr']:.7f}")
                print("-" * 70)

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE, scaler, writer,
                epoch, global_rank, ACCUM_STEPS
            )
            if global_rank == 0:
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, writer, epoch, global_rank)
            if global_rank == 0:
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.module.state_dict(), best_model_path)
                    print(f"✅ Best model saved with Val Acc: {val_acc:.2f}%")

            scheduler.step()

        # تست نهایی روی پوشهٔ Test
        if global_rank == 0:
            if os.path.exists(best_model_path):
                model.module.load_state_dict(torch.load(best_model_path))
            else:
                print("⚠️ Using last epoch weights for test.")

        test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, writer, NUM_EPOCHS, global_rank)

        if global_rank == 0:
            print("\n" + "=" * 70)
            print("🧪 Final Test on Clean (Non-overlapping) Test Set")
            print("=" * 70)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

            model_inference = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])
            model_inference.load_state_dict(model.module.state_dict())
            model_inference = model_inference.cpu().eval()

            checkpoint_inference = {
                'model_state_dict': model_inference.state_dict(),
                'total_params': sum(p.numel() for p in model_inference.parameters()),
                'masks': checkpoint['masks'],
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'training_config': {
                    'dataset': '200k-real-vs-ai-visuals (non-overlapping test)',
                    'lr': BASE_LR,
                    'weight_decay': WEIGHT_DECAY,
                    'batch_size': BATCH_SIZE,
                    'accum_steps': ACCUM_STEPS,
                    'epochs': NUM_EPOCHS,
                    'loss': 'BCEWithLogitsLoss'
                }
            }

            torch.save(checkpoint_inference, '/kaggle/working/final_pruned_finetuned_200k_inference.pt')
            print("✅ Inference-ready model saved!")

            if writer:
                writer.close()

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--accum_steps', type=int, default=1)
    args = parser.parse_args()
    main(args)
