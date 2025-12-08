import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.io as io
from torchvision.transforms import functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torchinfo import summary  # فرض کنید torchinfo نصب شده است
# اگر ptflops و thop نصب شده باشند، می‌توانید از آن‌ها استفاده کنید؛ در غیر این صورت کامنت کنید

# کلاس دیتاست برای ویدیوها
class VideoDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, num_frames=32, fps=30):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []
        
        # اسکن فولدرها برای ویدیوها (فرض: فرمت mp4)
        for label_name in ['Fake', 'Real']:
            label = 0 if label_name == 'Fake' else 1
            label_dir = os.path.join(root_dir, split, label_name)
            if os.path.exists(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith('.mp4'):
                        video_path = os.path.join(label_dir, file)
                        self.samples.append((video_path, label))
        
        self.samples_df = pd.DataFrame(self.samples, columns=['video_path', 'label'])
        print(f"{len(self.split)} dataset statistics:")
        print(f"Sample {self.split.lower()} video paths:")
        print(self.samples_df['video_path'].head())
        print(f"Total {self.split.lower()} dataset size: {len(self.samples_df)}")
        print(f"{self.split} label distribution:")
        print(self.samples_df['label'].value_counts().rename(index={0: 'Fake', 1: 'Real'}))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # خواندن ویدیو
        video, _, info = io.read_video(video_path, pts_unit='sec')
        if len(video) == 0:
            # اگر ویدیو خالی باشد، یک ویدیو dummy بسازید (برای سادگی)
            video = torch.zeros(self.num_frames, 240, 320, 3).byte()
        
        # نمونه‌برداری یکنواخت از فریم‌ها (uniform sampling)
        total_frames = len(video)
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = video[indices].float().permute(0, 3, 1, 2) / 255.0  # T C H W
        
        # تغییر اندازه به 256x256
        frames = F.interpolate(frames, size=(256, 256), mode='bilinear', align_corners=False)
        
        # نرمال‌سازی (ImageNet stats)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        frames = normalize(frames)  # اعمال روی همه فریم‌ها
        
        label = torch.tensor(label, dtype=torch.float32)
        return frames, label

# تابع collate برای stack کردن فریم‌ها
def collate_fn(batch):
    frames, labels = zip(*batch)
    frames = torch.stack(frames)  # B T C H W
    labels = torch.stack(labels)
    return frames, labels

# مدل ResNet50 برای ویدیو (frame-level aggregation)
class VideoResNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(VideoResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: B T C H W
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # (B*T) C H W
        feats = self.resnet.fc[: -1](x)  # ویژگی‌ها بدون fc نهایی: B*T 2048
        feats = feats.view(B, T, -1)  # B T 2048
        feats = feats.mean(dim=1)  # میانگین روی T: B 2048
        out = self.resnet.fc(feats)  # B 1
        out = self.sigmoid(out)  # برای BCE
        return out

# تابع آموزش
def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, device='cuda'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device).unsqueeze(1)
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet50_video.pth')
            print(f"Saved best resnet50 model with validation accuracy: {best_val_acc:.2f}% at epoch {epoch+1}")
    
    # ذخیره مدل نهایی
    torch.save(model.state_dict(), 'final_resnet50_video.pth')
    print("Saved final resnet50 model at epoch {}".format(num_epochs))
    
    return model

# تابع تست
def test_model(model, test_loader, device='cuda'):
    model.eval()
    criterion = nn.BCELoss()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device).unsqueeze(1)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    test_acc = 100 * test_correct / test_total
    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # نمایش confusion matrix (اختیاری)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    
    return test_acc

# اجرای اصلی
if __name__ == "__main__":
    # تنظیمات
    root_dir = '/path/to/your/dataset'  # مسیر دیتاست خود را وارد کنید
    batch_size = 8  # کوچکتر برای ویدیوها به دلیل حافظه (با num_frames=32)
    num_frames = 32  # تعداد فریم‌های نمونه‌برداری شده (می‌توانید تنظیم کنید، مثلاً 16 برای سرعت بیشتر)
    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ترانسفورم (فقط نرمال‌سازی، resize در دیتاست انجام می‌شود)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # دیتاست‌ها
    train_dataset = VideoDataset(root_dir, 'Train', transform=normalize, num_frames=num_frames)
    val_dataset = VideoDataset(root_dir, 'Validation', transform=normalize, num_frames=num_frames)
    test_dataset = VideoDataset(root_dir, 'Test', transform=normalize, num_frames=num_frames)
    
    # DataLoaderها
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Validation loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    # نمونه batch
    sample_train_batch = next(iter(train_loader))
    print(f"Sample train batch image shape: {sample_train_batch[0].shape}")  # [B, T, C, H, W]
    print(f"Sample train batch labels: {sample_train_batch[1]}")
    
    sample_val_batch = next(iter(val_loader))
    print(f"Sample validation batch image shape: {sample_val_batch[0].shape}")
    print(f"Sample validation batch labels: {sample_val_batch[1]}")
    
    sample_test_batch = next(iter(test_loader))
    print(f"Sample test batch image shape: {sample_test_batch[0].shape}")
    print(f"Sample test batch labels: {sample_test_batch[1]}")
    
    # مدل
    model = VideoResNet50(num_classes=1).to(device)
    
    # دانلود خودکار pretrained weights
    print("Downloading pretrained ResNet50...")
    
    # آموزش
    model = train_model(model, train_loader, val_loader, num_epochs, device=device)
    
    # تست
    test_acc = test_model(model, test_loader, device=device)
    
    # خلاصه مدل (با torchinfo، اگر نصب نباشد کامنت کنید)
    try:
        summary(model.resnet, input_size=(batch_size * num_frames, 3, 256, 256))
    except:
        print("torchinfo not available. Install with: pip install torchinfo")
    
    sample_frame = sample_train_batch[0][0, 0].detach().cpu()  # اولین فریم از اولین batch
    sample_frame = (sample_frame - sample_frame.min()) / (sample_frame.max() - sample_frame.min())  # normalize for viz
    plt.imshow(sample_frame.permute(1, 2, 0))
    plt.title("Sample Frame from Video")
    plt.show()
