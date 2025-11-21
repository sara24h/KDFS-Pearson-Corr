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

parser = argparse.ArgumentParser(description="Fine-tuning a Pruned ResNet Model for Deepfake Detection (190K pruned version)")

parser.add_argument('--data_dir', type=str, default='/kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k', help='Path to the dataset directory')
parser.add_argument('--pruned_model_path', type=str, default='/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt', help='Path to the pruned model checkpoint')
parser.add_argument('--output_path', type=str, default='/kaggle/working/best_model_190k.pth', help='Path to save the fine-tuned model')

# Training hyperparameters
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=15, help='Maximum number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability')
parser.add_argument('--patience', type=int, default=5, help='Patience for Early Stopping')

# Normalization stats
parser.add_argument('--mean', type=float, nargs='+', default=[0.4868,0.3972,0.3624], help='Mean values for normalization')
parser.add_argument('--std', type=float, nargs='+', default=[0.2296,0.2066,0.2009], help='Std values for normalization')

# Freezing strategy
parser.add_argument('--freeze_strategy', type=str, default='up_to_l4', choices=['none', 'fc_only', 'up_to_l3', 'up_to_l4'],
                    help="Strategy for freezing layers")

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Hyperparameters: {vars(args)}")

# =============================================================================
# Step 2: Define Pruned ResNet50 Architecture (190K filters)
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
        # Adjust channel mismatch due to pruning
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

        # New remaining filters based on your 190K model
        remaining_filters = [
            # layer1
            15, 9, 69,
            11, 9, 70,
            7, 4, 39,
            # layer2
            26, 28, 93,
            29, 14, 103,
            25, 12, 72,
            20, 13, 58,
            # layer3
            38, 30, 132,
            59, 21, 134,
            35, 13, 107,
            29, 10, 95,
            24, 8, 71,
            29, 8, 79,
            # layer4
            21, 21, 111,
            62, 4, 137,
            44, 3, 108,
        ]
        self.filter_iter = iter(remaining_filters)

        # Define layer output channels (as in original ResNet-50)
        layer1_out = 256   # 64 * 4
        layer2_out = 512   # 128 * 4
        layer3_out = 1024  # 256 * 4
        layer4_out = 2048  # 512 * 4

        self.layer1 = self._make_layer(block, layers[0], layer1_out, stride=1)
        self.layer2 = self._make_layer(block, layers[1], layer2_out, stride=2)
        self.layer3 = self._make_layer(block, layers[2], layer3_out, stride=2)
        self.layer4 = self._make_layer(block, layers[3], layer4_out, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer4_out, num_classes)

    def _make_layer(self, block, blocks, layer_out_channels, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != layer_out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, layer_out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(layer_out_channels)
            )

        layers = []
        conv1_c, conv2_c, conv3_c = next(self.filter_iter), next(self.filter_iter), next(self.filter_iter)
        layers.append(block(self.in_channels, layer_out_channels, conv1_c, conv2_c, conv3_c, stride, downsample))
        self.in_channels = layer_out_channels

        for _ in range(1, blocks):
            conv1_c, conv2_c, conv3_c = next(self.filter_iter), next(self.filter_iter), next(self.filter_iter)
            layers.append(block(self.in_channels, layer_out_channels, conv1_c, conv2_c, conv3_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.dropout(x); x = self.fc(x)
        return x

# =============================================================================
# Step 3: Data Preparation (resize to 224x224 for compatibility with pretrained weights)
# =============================================================================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=args.mean, std=args.std)
])
test_val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=args.mean, std=args.std)
])

train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transforms)
valid_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'valid'), transform=test_val_transforms)
test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=test_val_transforms)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Classes found: {train_dataset.classes}")
print(f"Train/Val/Test samples: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}")

model = ResNet_pruned(Bottleneck_pruned, [3, 4, 6, 3], dropout_prob=args.dropout_prob)

checkpoint = torch.load(args.pruned_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(DEVICE)
print("✅ Pruned model (190K) loaded successfully.")

# Extract masks if present
masks = None
if 'masks' in checkpoint:
    masks = checkpoint['masks']
    print("✅ Masks extracted from checkpoint.")
else:
    print("⚠️ No 'masks' key in checkpoint.")

# Freeze layers according to strategy
if args.freeze_strategy == 'fc_only':
    layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
elif args.freeze_strategy == 'up_to_l3':
    layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
elif args.freeze_strategy == 'up_to_l4':
    layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
else:
    layers_to_freeze = []

for name, param in model.named_parameters():
    if any(name.startswith(layer) for layer in layers_to_freeze):
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nFreezing strategy: '{args.freeze_strategy}' → Trainable params: {trainable_params}")

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(args.num_epochs):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        pred = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (pred == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = 100 * correct_train / total_train

    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            pred = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (pred == labels).sum().item()

    val_loss = val_loss / len(valid_dataset)
    val_acc = 100 * correct_val / total_val
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{args.num_epochs} → Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_checkpoint = {
            'model_state_dict': model.state_dict(),
            'hyperparameters': vars(args),
            'class_to_idx': train_dataset.class_to_idx
        }
        if masks is not None:
            best_checkpoint['masks'] = masks
        torch.save(best_checkpoint, args.output_path)
        print(f"  → New best model saved (Val Acc: {best_val_acc:.2f}%)")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  → No improvement ({epochs_no_improve}/{args.patience})")

    if epochs_no_improve >= args.patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print("\n✅ Training finished.")

# =============================================================================
# Step 6: Final Evaluation on Test Set
# =============================================================================
model.load_state_dict(torch.load(args.output_path, map_location=DEVICE)['model_state_dict'])
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="[Test Evaluation]"):
        images = images.to(DEVICE)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = accuracy_score(all_labels, all_preds) * 100
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "="*50)
print("         Final Test Results")
print("="*50)
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("="*50)
