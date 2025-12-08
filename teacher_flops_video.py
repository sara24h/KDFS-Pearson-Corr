import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import random
from pathlib import Path
from torchvision import transforms, models
import time
# فرض می‌کنیم این فایل در ساختار پروژه شما قرار دارد
from data.video_data import create_uadfv_dataloaders
from torchinfo import summary
from thop import profile
# اضافه کردن کتابخانه argparse
import argparse

def create_model(num_classes=1):
    """
    یک مدل ResNet50 از پیش آموزش دیده را بارگذاری کرده و برای Fine-tuning آماده می‌کند.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # فریز کردن تمام پارامترهای مدل
    for param in model.parameters():
        param.requires_grad = False

    # جایگزینی لایه آخر (classifier) برای تعداد کلاس‌های مورد نظر
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # unfreeze کردن لایه‌های آخر برای آموزش (Partial Fine-tuning)
    # شما می‌توانید لایه‌های بیشتری را unfreeze کنید
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for videos, labels in dataloader:
        videos = videos.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        batch_size, num_frames, C, H, W = videos.shape
        inputs = videos.view(batch_size * num_frames, C, H, W)
        
        outputs = model(inputs)
        outputs = outputs.view(batch_size, num_frames, -1).mean(dim=1)

        loss = criterion(outputs, labels)
        
        # --- اصلاح شده: محاسبه صحیح پیش‌بینی‌ها ---
        # خروجی مدل (logit) را با آستانه 0 مقایسه می‌کنیم
        preds = (outputs > 0).float()
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        # --- اصلاح شده: محاسبه صحیح تعداد پیش‌بینی‌های درست ---
        # مطمئن می‌شویم که شکل preds و labels یکسان است
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += videos.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device).unsqueeze(1)

            batch_size, num_frames, C, H, W = videos.shape
            inputs = videos.view(batch_size * num_frames, C, H, W)
            
            outputs = model(inputs)
            outputs = outputs.view(batch_size, num_frames, -1).mean(dim=1)
            
            loss = criterion(outputs, labels)
            
            # --- اصلاح شده: محاسبه صحیح پیش‌بینی‌ها ---
            preds = (outputs > 0).float()

            running_loss += loss.item() * videos.size(0)
            # --- اصلاح شده: محاسبه صحیح تعداد پیش‌بینی‌های درست ---
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += videos.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

# ==============================================================================
# بخش 3: اجرای اصلی (با تغییرات)
# ==============================================================================

if __name__ == "__main__":
    # --- بخش اضافه شده برای پارس کردن آرگومان‌های خط فرمان ---
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 on UADFV video dataset.")

    # اضافه کردن آرگومان‌ها با نوع، مقدار پیش‌فرض و توضیحات
    parser.add_argument('--root_dir', type=str, default="/kaggle/input/uadfv-dataset/UADFV", 
                        help='Root directory of the dataset.')
    parser.add_argument('--num_frames', type=int, default=32, 
                        help='Number of frames to sample from each video.')
    parser.add_argument('--image_size', type=int, default=256, 
                        help='Size of the input images.')
    parser.add_argument('--train_batch_size', type=int, default=4, 
                        help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=8, 
                        help='Batch size for validation and testing.')
    parser.add_argument('--num_epochs', type=int, default=15, 
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Initial learning rate.')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of worker processes for data loading.')

    # پارس کردن آرگومان‌های ورودی
    args = parser.parse_args()

    # --- استفاده از آرگومان‌های دریافتی ---
    print("--- Running with the following arguments ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-------------------------------------------")

    # --- تنظیم دستگاه (GPU/CPU) ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- ایجاد Dataloaderها ---
    train_loader, val_loader, test_loader = create_uadfv_dataloaders(
        root_dir=args.root_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        ddp=False,
        sampling_strategy='uniform'
    )

    # --- ایجاد مدل ---
    model = create_model(num_classes=1)
    model = model.to(device)

    # --- چاپ پیچیدگی مدل (تعداد پارامتر و FLOPs) ---
    print("\n" + "="*50)
    print("Model Architecture Summary (torchinfo):")
    print("="*50)
    summary(model, input_size=(1, 3, args.image_size, args.image_size), device=device)
    
    print("\n" + "="*50)
    print("Model Complexity (thop):")
    print("="*50)
    dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    print(f"Model FLOPs (thop): {flops/1e9:.2f} GFLOPs")
    print(f"Model Parameters (thop): {params/1e6:.2f} M")
    print("="*50 + "\n")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # --- حلقه آموزش ---
    best_val_acc = 0.0
    best_model_wts = None

    since = time.time()
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch + 1}/{args.num_epochs}')
        print('-' * 10)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # فراخوانی scheduler با مقدار loss روی مجموعه اعتبارسنجی
        scheduler.step(val_loss)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}')
        print('-' * 20)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, 'best_resnet50_partial_finetuned.pth')
            print(f"Saved best model with validation accuracy: {best_val_acc:.4f} at epoch {epoch + 1}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_val_acc:4f}')

    # --- بارگذاری بهترین مدل و ارزیابی روی مجموعه تست ---
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    
    print("\nEvaluating on the test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
