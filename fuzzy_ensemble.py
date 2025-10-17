"""
Fuzzy Ensemble for Binary Classification - Standalone Version
این نسخه شامل تمام کدهای لازم است و نیازی به فایل جداگانه ندارد
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# ============================================================
# بخش 1: معماری مدل Pruned ResNet
# ============================================================

def get_preserved_filter_num(mask):
    return int(mask.sum())


class BasicBlock_pruned(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, masks=[], stride=1):
        super().__init__()
        self.masks = masks

        preserved_filter_num1 = get_preserved_filter_num(masks[0])
        self.conv1 = nn.Conv2d(
            in_planes,
            preserved_filter_num1,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(preserved_filter_num1)
        preserved_filter_num2 = get_preserved_filter_num(masks[1])
        self.conv2 = nn.Conv2d(
            preserved_filter_num1,
            preserved_filter_num2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(preserved_filter_num2)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        shortcut_out = self.downsample(x)
        padded_out = torch.zeros_like(shortcut_out)

        idx = torch.nonzero(self.masks[1], as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            temp_full = torch.zeros_like(padded_out)
            for i, ch_idx in enumerate(idx):
                temp_full[:, ch_idx, :, :] = out[:, i, :, :]
            padded_out = temp_full

        assert padded_out.shape == shortcut_out.shape, "wrong shape"

        padded_out += shortcut_out
        padded_out = F.relu(padded_out)
        return padded_out


class Bottleneck_pruned(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, masks=[], stride=1):
        super().__init__()
        self.masks = masks

        preserved_filter_num1 = get_preserved_filter_num(masks[0])
        self.conv1 = nn.Conv2d(
            in_planes, preserved_filter_num1, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(preserved_filter_num1)
        preserved_filter_num2 = get_preserved_filter_num(masks[1])
        self.conv2 = nn.Conv2d(
            preserved_filter_num1,
            preserved_filter_num2,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(preserved_filter_num2)
        preserved_filter_num3 = get_preserved_filter_num(masks[2])
        self.conv3 = nn.Conv2d(
            preserved_filter_num2,
            preserved_filter_num3,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(preserved_filter_num3)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        shortcut_out = self.downsample(x)
        padded_out = torch.zeros_like(shortcut_out)

        idx = torch.nonzero(self.masks[2], as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            temp_full = torch.zeros_like(padded_out)
            for i, ch_idx in enumerate(idx):
                temp_full[:, ch_idx, :, :] = out[:, i, :, :]
            padded_out = temp_full

        assert padded_out.shape == shortcut_out.shape, "wrong shape"

        padded_out += shortcut_out
        padded_out = F.relu(padded_out)
        return padded_out


class ResNet_pruned(nn.Module):
    def __init__(self, block, num_blocks, masks=[], num_classes=1):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        coef = 0
        if block == BasicBlock_pruned:
            coef = 2
        elif block == Bottleneck_pruned:
            coef = 3
        num = 0
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            masks=masks[0 : coef * num_blocks[0]],
        )
        num = num + coef * num_blocks[0]

        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            masks=masks[num : num + coef * num_blocks[1]],
        )
        num = num + coef * num_blocks[1]

        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            masks=masks[num : num + coef * num_blocks[2]],
        )
        num = num + coef * num_blocks[2]

        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            masks=masks[num : num + coef * num_blocks[3]],
        )
        num = num + coef * num_blocks[3]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, masks=[]):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        coef = 0
        if block == BasicBlock_pruned:
            coef = 2
        elif block == Bottleneck_pruned:
            coef = 3

        for i, stride in enumerate(strides):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    masks[coef * i : coef * i + coef],
                    stride,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        for block in self.layer1:
            out = block(out)
        feature_list.append(out)

        for block in self.layer2:
            out = block(out)
        feature_list.append(out)

        for block in self.layer3:
            out = block(out)
        feature_list.append(out)

        for block in self.layer4:
            out = block(out)
        feature_list.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list


def ResNet_50_pruned_hardfakevsreal(masks): 
    return ResNet_pruned(
        block=Bottleneck_pruned, num_blocks=[3, 4, 6, 3], masks=masks, num_classes=1
    )


# ============================================================
# بخش 2: Dataset و توابع کمکی
# ============================================================

class WildDeepfakeDataset(Dataset):
    """دیتاست سفارشی برای Wild-Deepfake"""
    def __init__(self, real_path, fake_path, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # خواندن تصاویر real (label=0)
        if os.path.exists(real_path):
            real_files = [f for f in os.listdir(real_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in real_files:
                self.images.append(os.path.join(real_path, fname))
                self.labels.append(0)
        
        # خواندن تصاویر fake (label=1)
        if os.path.exists(fake_path):
            fake_files = [f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in fake_files:
                self.images.append(os.path.join(fake_path, fname))
                self.labels.append(1)
        
        print(f"تعداد تصاویر Real: {len([l for l in self.labels if l==0])}")
        print(f"تعداد تصاویر Fake: {len([l for l in self.labels if l==1])}")
        print(f"مجموع تصاویر: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            print(f"❌ خطا در لود {img_path}: {e}")
            return torch.zeros(3, 224, 224), label, img_path


def load_pruned_model(checkpoint_path, device):
    """لود کردن مدل pruned از checkpoint"""
    print(f"🔄 در حال لود مدل از: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        print(f"✅ Checkpoint شامل کلیدهای: {list(checkpoint.keys())}")
        
        # استخراج masks
        masks = checkpoint.get('masks', None)
        if masks is not None:
            masks_detached = [m.detach().clone() if m is not None else None for m in masks]
        else:
            masks_detached = None
            print("⚠️ هیچ mask در checkpoint یافت نشد")
        
        # ساخت مدل با masks
        model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
        
        # لود کردن وزن‌ها
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ وزن‌ها از 'model_state_dict' لود شدند")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("✅ وزن‌ها از 'state_dict' لود شدند")
        else:
            model.load_state_dict(checkpoint)
            print("✅ وزن‌ها مستقیماً از checkpoint لود شدند")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 تعداد کل پارامترها: {total_params:,}")
        
    else:
        raise ValueError("❌ فرمت checkpoint نامعتبر است!")
    
    model = model.to(device)
    model.eval()
    
    return model


def get_predictions(model, dataloader, device):
    """پیش‌بینی احتمالات برای دسته‌بندی باینری"""
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Getting predictions"):
            images = images.to(device)
            
            # پیش‌بینی (مدل یک خروجی و feature map برمی‌گرداند)
            outputs, _ = model(images)
            
            # تبدیل logits به احتمالات
            probs_fake = torch.sigmoid(outputs).squeeze()
            
            if probs_fake.dim() == 0:
                probs_fake = probs_fake.unsqueeze(0)
            
            probs_real = 1 - probs_fake
            probs_2class = torch.stack([probs_real, probs_fake], dim=1)
            
            all_probs.append(probs_2class.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    return all_probs, all_labels


# ============================================================
# بخش 3: توابع Fuzzy Ensemble
# ============================================================

def generateRank1(score, class_no=2):
    """تابع رتبه‌بندی فازی اول"""
    rank = np.zeros([class_no, 1])
    scores = score.reshape(-1, 1)
    
    for i in range(class_no):
        rank[i] = 1 - np.exp(-((scores[i] - 1) ** 2) / 2.0)
    
    return rank


def generateRank2(score, class_no=2):
    """تابع رتبه‌بندی فازی دوم"""
    rank = np.zeros([class_no, 1])
    scores = score.reshape(-1, 1)
    
    for i in range(class_no):
        rank[i] = 1 - np.tanh(((scores[i] - 1) ** 2) / 2)
    
    return rank


def fuzzy_ensemble_binary(res1, res2, labels, class_no=2):
    """ترکیب فازی برای دسته‌بندی باینری"""
    correct = 0
    predictions = []
    fusion_details = []
    
    for i in range(len(res1)):
        # محاسبه رتبه‌های فازی
        rank1 = generateRank1(res1[i], class_no) * generateRank2(res1[i], class_no)
        rank2 = generateRank1(res2[i], class_no) * generateRank2(res2[i], class_no)
        
        # جمع رتبه‌ها
        rankSum = rank1 + rank2
        rankSum = np.array(rankSum)
        
        # محاسبه میانگین امتیازات
        scoreSum = 1 - (res1[i] + res2[i]) / 2
        scoreSum = np.array(scoreSum).reshape(-1, 1)
        
        # محاسبه امتیاز نهایی فازی
        fusedScore = (rankSum.T) * scoreSum
        
        # انتخاب کلاس با کمترین رتبه
        cls = np.argmin(rankSum)
        predictions.append(cls)
        
        # ذخیره جزئیات
        fusion_details.append({
            'sample_idx': i,
            'model1_probs': res1[i],
            'model2_probs': res2[i],
            'rank1': rank1.flatten(),
            'rank2': rank2.flatten(),
            'rankSum': rankSum.flatten(),
            'fusedScore': fusedScore.flatten(),
            'prediction': cls,
            'true_label': labels[i]
        })
        
        if cls < class_no and labels[i] == cls:
            correct += 1
    
    accuracy = correct / len(res1)
    
    return np.array(predictions), accuracy, fusion_details


def print_detailed_results(labels, predictions, model1_probs, model2_probs):
    """نمایش نتایج تفصیلی"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "="*70)
    print("📊 گزارش دسته‌بندی:")
    print("="*70)
    print(classification_report(labels, predictions, 
                                target_names=['Real', 'Fake'],
                                digits=4))
    
    print("\n" + "="*70)
    print("📈 ماتریس درهم‌ریختگی:")
    print("="*70)
    cm = confusion_matrix(labels, predictions)
    print(f"\n{'':15} {'Predicted Real':>15} {'Predicted Fake':>15}")
    print(f"{'Actual Real':15} {cm[0,0]:>15} {cm[0,1]:>15}")
    print(f"{'Actual Fake':15} {cm[1,0]:>15} {cm[1,1]:>15}")
    
    print("\n" + "="*70)
    print("📊 آمار تفصیلی:")
    print("="*70)
    print(f"✅ Real correctly classified: {cm[0,0]} / {cm[0,0] + cm[0,1]}")
    print(f"❌ Real misclassified as Fake: {cm[0,1]} / {cm[0,0] + cm[0,1]}")
    print(f"✅ Fake correctly classified: {cm[1,1]} / {cm[1,0] + cm[1,1]}")
    print(f"❌ Fake misclassified as Real: {cm[1,0]} / {cm[1,0] + cm[1,1]}")
    
    # محاسبه دقت هر مدل
    model1_preds = np.argmax(model1_probs, axis=1)
    model2_preds = np.argmax(model2_probs, axis=1)
    
    model1_acc = (model1_preds == labels).sum() / len(labels)
    model2_acc = (model2_preds == labels).sum() / len(labels)
    ensemble_acc = (predictions == labels).sum() / len(labels)
    
    print("\n" + "="*70)
    print("🔬 مقایسه عملکرد مدل‌ها:")
    print("="*70)
    print(f"Model 1 Accuracy: {model1_acc*100:.2f}%")
    print(f"Model 2 Accuracy: {model2_acc*100:.2f}%")
    print(f"Fuzzy Ensemble Accuracy: {ensemble_acc*100:.2f}%")
    improvement = (ensemble_acc - max(model1_acc, model2_acc))*100
    print(f"بهبود نسبت به بهترین مدل: {improvement:+.2f}%")


# ============================================================
# بخش 4: Main Function
# ============================================================

def main():
    # تنظیمات
    BATCH_SIZE = 32
    IMG_SIZE = 224
    NUM_WORKERS = 4
    
    # تنظیم دستگاه
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 دستگاه: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 حافظه GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # تعریف transforms
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4414, 0.3448, 0.3159], 
                           std=[0.1854, 0.1623, 0.1562])
    ])
    
    # ایجاد dataset و dataloader
    print("\n📂 در حال بارگذاری دیتاست...")
    test_dataset = WildDeepfakeDataset(
        real_path="/kaggle/input/wild-deepfake/test/real",
        fake_path="/kaggle/input/wild-deepfake/test/fake",
        transform=val_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # لود کردن مدل‌ها
    print("\n🔄 در حال بارگذاری مدل‌ها...")
    
    MODEL1_PATH = "/kaggle/input/10k_finetune_wd/pytorch/default/1/10k_final_pruned_finetuned_inference_ready (1).pt"
    MODEL2_PATH = "/kaggle/input/140k_finetuned_wd/pytorch/default/1/140k_final_pruned_finetuned_inference_ready (1).pt"
    
    model1 = load_pruned_model(MODEL1_PATH, device)
    print("✅ Model 1 لود شد\n")
    
    model2 = load_pruned_model(MODEL2_PATH, device)
    print("✅ Model 2 لود شد\n")
    
    # دریافت پیش‌بینی‌ها
    print("🔮 Model 1: در حال پیش‌بینی...")
    predictions1, labels = get_predictions(model1, test_loader, device)
    
    print("🔮 Model 2: در حال پیش‌بینی...")
    predictions2, _ = get_predictions(model2, test_loader, device)
    
    print(f"\n✅ پیش‌بینی‌ها دریافت شد")
    print(f"   - تعداد نمونه‌ها: {len(predictions1)}")
    print(f"   - شکل احتمالات: {predictions1.shape}")
    
    # ترکیب فازی
    print("\n" + "="*70)
    print("🎯 در حال ترکیب فازی مدل‌ها...")
    print("="*70)
    
    final_predictions, accuracy, fusion_details = fuzzy_ensemble_binary(
        predictions1, 
        predictions2, 
        labels,
        class_no=2
    )
    
    print(f"\n✅ دقت ترکیب فازی: {accuracy * 100:.2f}%")
    print(f"✅ تعداد پیش‌بینی صحیح: {int(accuracy * len(labels))}/{len(labels)}")
    
    # نمایش نتایج تفصیلی
    print_detailed_results(labels, final_predictions, predictions1, predictions2)
    
    # ذخیره نتایج
    print("\n💾 در حال ذخیره نتایج...")
    
    results = {
        'final_predictions': final_predictions,
        'true_labels': labels,
        'accuracy': accuracy,
        'model1_probabilities': predictions1,
        'model2_probabilities': predictions2,
        'fusion_details': fusion_details[:100],
        'dataset_info': {
            'total_samples': len(labels),
            'real_samples': int((labels == 0).sum()),
            'fake_samples': int((labels == 1).sum())
        }
    }
    
    torch.save(results, 'fuzzy_ensemble_results.pt')
    print("✅ نتایج در 'fuzzy_ensemble_results.pt' ذخیره شد")
    
    # ذخیره CSV
    import pandas as pd
    
    df_results = pd.DataFrame({
        'true_label': labels,
        'fuzzy_prediction': final_predictions,
        'model1_prob_real': predictions1[:, 0],
        'model1_prob_fake': predictions1[:, 1],
        'model2_prob_real': predictions2[:, 0],
        'model2_prob_fake': predictions2[:, 1],
        'is_correct': (final_predictions == labels).astype(int)
    })
    
    df_results.to_csv('fuzzy_ensemble_results.csv', index=False)
    print("✅ نتایج در 'fuzzy_ensemble_results.csv' ذخیره شد")
    
    print("\n" + "="*70)
    print("🎉 پردازش با موفقیت انجام شد!")
    print("="*70)


if __name__ == "__main__":
    main()
