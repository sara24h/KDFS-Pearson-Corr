import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch.nn.functional as F
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Fine-tuning 190k Pruned ResNet50 for Deepfake Detection")
parser.add_argument('--data_dir', type=str, default='/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k')
parser.add_argument('--pruned_model_path', type=str, default='/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt')
parser.add_argument('--output_path', type=str, default='/kaggle/working/best_190k_finetuned.pth')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--dropout_prob', type=float, default=0.5)
parser.add_argument('--patience', type=int, default=6)
parser.add_argument('--freeze_strategy', type=str, default='fc_only', 
                    choices=['none', 'fc_only', 'up_to_l4', 'up_to_l3'], help="فریز کردن لایه‌ها")

# نرمال‌سازی دقیق دیتاست Wild-Deepfake
parser.add_argument('--mean', type=float, nargs='+', default=[0.5207, 0.4258, 0.3806])
parser.add_argument('--std', type=float, nargs='+', default=[0.2490, 0.2239, 0.2212])

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Hyperparameters: {vars(args)}")

# =============================================================================
# معماری دقیق مدل هرس‌شده 190k (بر اساس خروجی شما)
# =============================================================================
class Bottleneck_pruned(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, conv1_channels, conv2_channels, conv3_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.conv3 = nn.Conv2d(conv2_channels, conv3_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv3_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        # تطبیق کانال‌ها (مهم!)
        if out.shape[1] != identity.shape[1]:
            diff = identity.shape[1] - out.shape[1]
            if diff > 0:
                out = F.pad(out, [0, 0, 0, 0, 0, diff])
            else:
                identity = F.pad(identity, [0, 0, 0, 0, 0, -diff])

        out += identity
        return self.relu(out)


class ResNet_pruned(nn.Module):
    def __init__(self, block, layers, num_classes=1, dropout_prob=0.5):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # تعداد فیلترهای زنده در هر conv (دقیقاً از مدل 190k)
        self.remaining_filters = [
            # layer1 (3 blocks)
            15, 9, 69,    11, 9, 70,    7, 4, 39,
            # layer2 (4 blocks)
            26,28,93,  29,14,103,  25,12,72,  20,13,58,
            # layer3 (6 blocks)
            38,30,132,  59,21,134,  35,13,107,  29,10,95,  24,8,71,  29,8,79,
            # layer4 (3 blocks)
            21,21,111,  62,4,137,  44,3,108
        ]

        filter_iter = iter(self.remaining_filters)

        self.layer1 = self._make_layer(block, layers[0], filter_iter, 256, stride=1)
        self.layer2 = self._make_layer(block, layers[1], filter_iter, 512, stride=2)
        self.layer3 = self._make_layer(block, layers[2], filter_iter, 1024, stride=2)
        self.layer4 = self._make_layer(block, layers[3], filter_iter, 2048, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)  # همیشه 2048 چون downsample به 2048 میرسه
        self.dropout = nn.Dropout(dropout_prob)

    def _make_layer(self, block, blocks, filter_iter, out_channels, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        c1, c2, c3 = next(filter_iter), next(filter_iter), next(filter_iter)
        layers.append(block(self.in_channels, out_channels, c1, c2, c3, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            c1, c2, c3 = next(filter_iter), next(filter_iter), next(filter_iter)
            layers.append(block(self.in_channels, out_channels, c1, c2, c3))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# =============================================================================
# دیتالودرها
# =============================================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=args.mean, std=args.std),
])

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=args.mean, std=args.std),
])

train_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, 'valid'), transform=val_test_transform)
test_ds  = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# =============================================================================
# ساخت و لود مدل
# =============================================================================
model = ResNet_pruned(Bottleneck_pruned, [3, 4, 6, 3], dropout_prob=args.dropout_prob).to(DEVICE)

checkpoint = torch.load(args.pruned_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)  # یا اگر فقط وزن هست: checkpoint
model.to(DEVICE)
print("مدل هرس‌شده 190k با موفقیت لود شد")

# استخراج ماسک‌ها (اختیاری برای ذخیره بعدی)
try:
    masks = checkpoint['masks']
    print(f"ماسک‌ها استخراج شد: {len(masks)} تا")
except:
    masks = None
    print("ماسک در چک‌پوینت نبود")

# =============================================================================
# فریز کردن لایه‌ها
# =============================================================================
if args.freeze_strategy == 'fc_only':
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
elif args.freeze_strategy == 'up_to_l4':
    for name, param in model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
elif args.freeze_strategy == 'up_to_l3':
    for name, param in model.named_parameters():
        if not (name.startswith('layer4') or name.startswith('fc')):
            param.requires_grad = False
else:  # none
    pass

print("پارامترهای قابل آموزش:")
for n, p in model.named_parameters():
    if p.requires_grad:
        print(f"   {n}")

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

# =============================================================================
# حلقه آموزش
# =============================================================================
best_acc = 0.0
patience_counter = 0

for epoch in range(args.num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE, dtype=torch.float).unsqueeze(1)

        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = (torch.sigmoid(outs) > 0.5).float()
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE, dtype=torch.float).unsqueeze(1)
            outs = model(imgs)
            val_loss += criterion(outs, lbls).item()
            preds = (torch.sigmoid(outs) > 0.5).float()
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    val_acc = 100 * correct / total
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'epoch': epoch,
            'args': vars(args)
        }
        if masks is not None:
            save_dict['masks'] = masks
        torch.save(save_dict, args.output_path)
        print(f"   Best model saved: {best_acc:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print("Early stopping!")
            break

print("آموزش تمام شد. در حال تست نهایی...")

# =============================================================================
# تست نهایی
# =============================================================================
model.load_state_dict(torch.load(args.output_path)['model_state_dict'])
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, lbls in tqdm(test_loader, desc="Final Test"):
        imgs = imgs.to(DEVICE)
        outs = model(imgs)
        preds = (torch.sigmoid(outs) > 0.5).cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())

acc = accuracy_score(all_labels, all_preds) * 100
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

print("\n" + "="*60)
print(f"   Final Test Accuracy : {acc:.2f}%")
print(f"   Precision           : {prec:.4f}")
print(f"   Recall              : {rec:.4f}")
print(f"   F1-Score            : {f1:.4f}")
print("="*60)
