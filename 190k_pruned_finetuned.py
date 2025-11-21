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

parser = argparse.ArgumentParser(description="Fine-tuning a Pruned ResNet Model for Deepfake Detection")

parser.add_argument('--data_dir', type=str, default='/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k', help='Path to the dataset directory')
parser.add_argument('--pruned_model_path', type=str, default='/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt', help='Path to the pruned model checkpoint')
parser.add_argument('--output_path', type=str, default='/kaggle/working/best_model.pth', help='Path to save the fine-tuned model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=15, help='Maximum number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability')
parser.add_argument('--patience', type=int, default=5, help='Patience for Early Stopping')

# آرگومان‌های نرمال‌سازی
parser.add_argument('--mean', type=float, nargs='+', default=[0.5207, 0.4258, 0.3806], help='Mean values for normalization')
parser.add_argument('--std', type=float, nargs='+', default=[0.2490, 0.2239, 0.2212], help='Std values for normalization')

# آرگومان استراتژی انجماد لایه‌ها
parser.add_argument('--freeze_strategy', type=str, default='up_to_l4', choices=['none', 'fc_only', 'up_to_l3', 'up_to_l4'],
                    help="Strategy for freezing layers: 'none' (train all), 'fc_only' (freeze all but fc), 'up_to_l3' (freeze up to layer 3), 'up_to_l4' (freeze all but fc)")

args = parser.parse_args()

# استفاده از آرگومان‌های خوانده شده
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Hyperparameters: {vars(args)}")


# =============================================================================
# مرحله 2: تعریف معماری مدل هرس‌شده
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
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if out.shape[1] != identity.shape[1]:
            if out.shape[1] < identity.shape[1]:
                pad_channels = identity.shape[1] - out.shape[1]
                out = F.pad(out, (0, 0, 0, 0, 0, pad_channels))
            else:
                pad_channels = out.shape[1] - identity.shape[1]
                identity = F.pad(identity, (0, 0, 0, 0, 0, pad_channels))
        out += identity
        out = self.relu(out)
        return out

class ResNet_pruned(nn.Module):
    def __init__(self, block, layers, num_classes=1, dropout_prob=0.5):
        super(ResNet_pruned, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob)
        remaining_filters = [12, 12, 64, 11, 10, 59, 9, 11, 49, 23, 21, 86, 22, 17, 86, 23, 17, 66, 21, 14, 56, 20, 48, 105, 54, 14, 95, 35, 11, 77, 42, 12, 50, 41, 7, 35, 28, 3, 41, 37, 55, 55, 78, 5, 67, 52, 16, 89]
        filter_iter = iter(remaining_filters)
        layer1_out_channels = 64 * block.expansion
        layer2_out_channels = 128 * block.expansion
        layer3_out_channels = 256 * block.expansion
        layer4_out_channels = 512 * block.expansion
        self.layer1 = self._make_layer(block, layers[0], filter_iter, layer1_out_channels, stride=1)
        self.layer2 = self._make_layer(block, layers[1], filter_iter, layer2_out_channels, stride=2)
        self.layer3 = self._make_layer(block, layers[2], filter_iter, layer3_out_channels, stride=2)
        self.layer4 = self._make_layer(block, layers[3], filter_iter, layer4_out_channels, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer4_out_channels, num_classes)

    def _make_layer(self, block, blocks, filter_iter, layer_out_channels, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != layer_out_channels:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, layer_out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(layer_out_channels))
        layers = []
        conv1_c, conv2_c, conv3_c = next(filter_iter), next(filter_iter), next(filter_iter)
        layers.append(block(self.in_channels, layer_out_channels, conv1_c, conv2_c, conv3_c, stride, downsample))
        self.in_channels = layer_out_channels
        for _ in range(1, blocks):
            conv1_c, conv2_c, conv3_c = next(filter_iter), next(filter_iter), next(filter_iter)
            layers.append(block(self.in_channels, layer_out_channels, conv1_c, conv2_c, conv3_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.dropout(x); x = self.fc(x)
        return x

# =============================================================================
# مرحله 3: آماده‌سازی داده‌ها (DataLoaders)
# =============================================================================
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)), transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(), transforms.Normalize(mean=args.mean, std=args.std)
])
test_val_transforms = transforms.Compose([
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize(mean=args.mean, std=args.std)
])
train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transforms)
valid_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'valid'), transform=test_val_transforms)
test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=test_val_transforms)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Classes found: {train_dataset.classes}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# =============================================================================
# مرحله 4: بارگذاری مدل، انجماد لایه‌ها، تعریف تابع هزینه و بهینه‌ساز
# =============================================================================
model = ResNet_pruned(Bottleneck_pruned, [3, 4, 6, 3], dropout_prob=args.dropout_prob)
checkpoint = torch.load(args.pruned_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(DEVICE)
print("✅ مدل با موفقیت بارگذاری شد.")

# --- تغییر کلیدی ۱: استخراج ماسک‌ها از چک‌پوینت اصلی ---
try:
    masks = checkpoint['masks']
    print("✅ ماسک‌ها با موفقیت از چک‌پوینت اصلی استخراج شدند.")
except KeyError:
    print("⚠️  هشدار: کلید 'masks' در چک‌پوینت اصلی یافت نشد. ماسک‌ها ذخیره نخواهند شد.")
    masks = None # یا می‌توانید یک دیکشنری خالی ایجاد کنید

# --- منطق انجماد لایه‌ها بر اساس استراتژی انتخاب شده ---
if args.freeze_strategy == 'fc_only':
    layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
elif args.freeze_strategy == 'up_to_l3':
    layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
elif args.freeze_strategy == 'up_to_l4':
    layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
else: # 'none'
    layers_to_freeze = []

for name, param in model.named_parameters():
    if any(name.startswith(layer_name) for layer_name in layers_to_freeze):
        param.requires_grad = False

print(f"\nLayer freezing strategy: '{args.freeze_strategy}'")
print("وضعیت آموزشی پارامترها:")
for name, param in model.named_parameters():
    if param.requires_grad: print(f"{name}: Trainable")
print("-" * 30)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# =============================================================================
# مرحله 5: حلقه آموزش و اعتبارسنجی
# =============================================================================
best_val_acc = 0.0
epochs_no_improve = 0
for epoch in range(args.num_epochs):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Training]", leave=False):
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
    epoch_loss, epoch_acc = running_loss / len(train_dataset), 100 * correct_train / total_train

    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Validation]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    epoch_val_loss, epoch_val_acc = val_loss / len(valid_dataset), 100 * correct_val / total_val
    scheduler.step(epoch_val_loss)

    print(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
    
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc

        best_checkpoint = {
            'model_state_dict': model.state_dict(),
            'hyperparameters': vars(args),
            'class_to_idx': train_dataset.class_to_idx
        }
        if masks is not None:
            best_checkpoint['masks'] = masks
        
        torch.save(best_checkpoint, args.output_path)
        print(f"  -> New best model saved with validation accuracy: {best_val_acc:.2f}%")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  -> Validation accuracy did not improve. ({epochs_no_improve}/{args.patience})")
    if epochs_no_improve >= args.patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs."); break
print("\nFinished Training.")

model.load_state_dict(torch.load(args.output_path)['model_state_dict'])
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="[Final Testing]", leave=True):
        images = images.to(DEVICE)
        outputs = model(images)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
all_preds, all_labels = np.array(all_preds).flatten(), np.array(all_labels)
test_acc = accuracy_score(all_labels, all_preds) * 100
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)
print("\n" + "="*50); print("         Final Test Results"); print("="*50)
print(f"Test Accuracy: {test_acc:.2f}%"); print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}"); print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:"); print(cm); print("="*50)
