# =============================================================================
# مرحله 0: نصب کتابخانه‌های مورد نیاز (در صورت لزوم)
# =============================================================================
# !pip install torch torchvision scikit-learn tqdm

# =============================================================================
# مرحله 1: وارد کردن کتابخانه‌ها
# =============================================================================
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
import torch.nn.functional as F # برای عملیات پد کردن

warnings.filterwarnings("ignore")

# =============================================================================
# مرحله 2: تعریف معماری مدل هرس‌شده با منطق اتصال پرش اصلاح‌شده
# =============================================================================
class Bottleneck_pruned(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, conv1_channels, conv2_channels, conv3_channels, stride=1, downsample=None):
        super(Bottleneck_pruned, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.conv3 = nn.Conv2d(conv2_channels, conv3_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv3_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # *** بخش کلیدی برای حل مشکل عدم تطابق ابعاد ***
        # اگر تعداد کانال‌ها برابر نبود، تانسور کوچکتر را پد می‌کنیم
        if out.shape[1] != identity.shape[1]:
            if out.shape[1] < identity.shape[1]:
                # پد کردن 'out' تا هم‌اندازه 'identity' شود
                pad_channels = identity.shape[1] - out.shape[1]
                out = F.pad(out, (0, 0, 0, 0, 0, pad_channels))
            else:
                # پد کردن 'identity' تا هم‌اندازه 'out' شود
                pad_channels = out.shape[1] - identity.shape[1]
                identity = F.pad(identity, (0, 0, 0, 0, 0, pad_channels))
        
        out += identity
        out = self.relu(out)

        return out

class ResNet_pruned(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet_pruned, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        remaining_filters = [
            # layer1
            12, 12, 64, 11, 10, 59, 9, 11, 49,
            # layer2
            23, 21, 86, 22, 17, 86, 23, 17, 66, 21, 14, 56,
            # layer3
            20, 48, 105, 54, 14, 95, 35, 11, 77, 42, 12, 50, 41, 7, 35, 28, 3, 41,
            # layer4
            37, 55, 55, 78, 5, 67, 52, 16, 89
        ]
        filter_iter = iter(remaining_filters)

        # تعریف تعداد کانال‌های استاندارد برای هر لایه
        layer1_out_channels = 64 * block.expansion # 256
        layer2_out_channels = 128 * block.expansion # 512
        layer3_out_channels = 256 * block.expansion # 1024
        layer4_out_channels = 512 * block.expansion # 2048

        self.layer1 = self._make_layer(block, layers[0], filter_iter, layer1_out_channels, stride=1)
        self.layer2 = self._make_layer(block, layers[1], filter_iter, layer2_out_channels, stride=2)
        self.layer3 = self._make_layer(block, layers[2], filter_iter, layer3_out_channels, stride=2)
        self.layer4 = self._make_layer(block, layers[3], filter_iter, layer4_out_channels, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # لایه نهایی باید به تعداد کانال‌های خروجی آخرین لایه (که استاندارد است) متصل شود
        self.fc = nn.Linear(layer4_out_channels, num_classes)

    def _make_layer(self, block, blocks, filter_iter, layer_out_channels, stride=1):
        downsample = None
        # اگر استراید بزرگتر از ۱ باشد یا کانال‌های ورودی با خروجی لایه برابر نباشد، به downsample نیاز است
        if stride != 1 or self.in_channels != layer_out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, layer_out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(layer_out_channels),
            )

        layers = []
        # ساخت اولین بلوک در لایه
        conv1_c = next(filter_iter)
        conv2_c = next(filter_iter)
        conv3_c = next(filter_iter)
        layers.append(block(self.in_channels, layer_out_channels, conv1_c, conv2_c, conv3_c, stride, downsample))
        
        # به‌روزرسانی تعداد کانال‌های ورودی برای بلوک‌های بعدی به مقدار استاندارد لایه
        self.in_channels = layer_out_channels

        # ساخت بلوک‌های باقی‌مانده در لایه
        for _ in range(1, blocks):
            conv1_c = next(filter_iter)
            conv2_c = next(filter_iter)
            conv3_c = next(filter_iter)
            # برای بلوک‌های بعدی، ورودی برابر با خروجی استاندارد لایه است
            layers.append(block(self.in_channels, layer_out_channels, conv1_c, conv2_c, conv3_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# =============================================================================
# مرحله 3: تنظیمات اولیه و پارامترها
# =============================================================================
DATA_DIR = '/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k'
PRUNED_MODEL_PATH = '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt'
MEAN = [0.5207, 0.4258, 0.3806]
STD = [0.2490, 0.2239, 0.2212]
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# مرحله 4: آماده‌سازی داده‌ها (DataLoaders)
# =============================================================================
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

test_val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
valid_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), transform=test_val_transforms)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Classes found: {train_dataset.classes}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# =============================================================================
# مرحله 5: بارگذاری صحیح مدل، تعریف تابع هزینه و بهینه‌ساز
# =============================================================================
model = ResNet_pruned(Bottleneck_pruned, [3, 4, 6, 3])
checkpoint = torch.load(PRUNED_MODEL_PATH, map_location=DEVICE)

# با تعریف جدید، مدل باید بدون خطا و با strict=True بارگذاری شود
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(DEVICE)
print("✅ مدل با موفقیت و بدون خطا بارگذاری شد.")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", leave=False)
    for images, labels in train_progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * correct_train / total_train

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        val_progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", leave=False)
        for images, labels in val_progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(valid_dataset)
    epoch_val_acc = 100 * correct_val / total_val

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), '/kaggle/working/best_finetuned_pruned_model.pth')
        print(f"  -> New best model saved with validation accuracy: {best_val_acc:.2f}%")

print("\nFinished Training.")

model.load_state_dict(torch.load('/kaggle/working/best_finetuned_pruned_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    test_progress_bar = tqdm(test_loader, desc="[Final Testing]", leave=True)
    for images, labels in test_progress_bar:
        images = images.to(DEVICE)
        outputs = model(images)
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels)

test_acc = accuracy_score(all_labels, all_preds) * 100
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "="*50)
print("         Final Test Results on 20k-wild-dataset")
print("="*50)
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"F1-Score:     {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("="*50)
