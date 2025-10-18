import argparse
from model.pruned_model.Resnet_final import ResNet_50_pruned_hardfakevsreal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import pandas as pd


class WildDeepfakeDataset(Dataset):
    def __init__(self, real_path, fake_path, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        if os.path.exists(real_path):
            real_files = [f for f in os.listdir(real_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in real_files:
                self.images.append(os.path.join(real_path, fname))
                self.labels.append(0)
        else:
            raise FileNotFoundError(f"پوشه real یافت نشد: {real_path}")
        
        if os.path.exists(fake_path):
            fake_files = [f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in fake_files:
                self.images.append(os.path.join(fake_path, fname))
                self.labels.append(1)
        else:
            raise FileNotFoundError(f"پوشه fake یافت نشد: {fake_path}")
        
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
    print(f"🔄 در حال لود مدل از: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        masks = checkpoint.get('masks', None)
        if masks is not None:
            masks_detached = [m.detach().clone() if m is not None else None for m in masks]
        else:
            masks_detached = None
            print("⚠️ هیچ mask در checkpoint یافت نشد")
        
        model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
        
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
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Getting predictions"):
            images = images.to(device)
            outputs, _ = model(images)
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


def generateRank1(score, class_no=2):
    rank = np.zeros([class_no, 1])
    scores = score.reshape(-1, 1)
    for i in range(class_no):
        rank[i] = 1 - np.exp(-((scores[i] - 1) ** 2) / 2.0)
    return rank

def generateRank2(score, class_no=2):
    rank = np.zeros([class_no, 1])
    scores = score.reshape(-1, 1)
    for i in range(class_no):
        rank[i] = 1 - np.tanh(((scores[i] - 1) ** 2) / 2)
    return rank

def fuzzy_ensemble_binary(res1, res2, labels, class_no=2):
    correct = 0
    predictions = []
    fusion_details = []
    for i in range(len(res1)):
        rank1 = generateRank1(res1[i], class_no) * generateRank2(res1[i], class_no)
        rank2 = generateRank1(res2[i], class_no) * generateRank2(res2[i], class_no)
        rankSum = rank1 + rank2
        scoreSum = 1 - (res1[i] + res2[i]) / 2
        fusedScore = (rankSum.T) * scoreSum
        cls = np.argmin(rankSum)
        predictions.append(cls)
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
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n" + "="*70)
    print("📊 گزارش دسته‌بندی:")
    print("="*70)
    print(classification_report(labels, predictions, target_names=['Real', 'Fake'], digits=4))
    
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


def main():
    parser = argparse.ArgumentParser(description="Fuzzy Ensemble برای دو مدل هرس‌شده Deepfake")
    parser.add_argument('--model1_path', type=str, required=True, help='مسیر فایل مدل اول (.pt)')
    parser.add_argument('--model2_path', type=str, required=True, help='مسیر فایل مدل دوم (.pt)')
    parser.add_argument('--test_real_dir', type=str, required=True, help='مسیر پوشه تصاویر real تست')
    parser.add_argument('--test_fake_dir', type=str, required=True, help='مسیر پوشه تصاویر fake تست')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size برای تست')
    parser.add_argument('--num_workers', type=int, default=4, help='تعداد worker برای DataLoader')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 دستگاه: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4414, 0.3448, 0.3159], std=[0.1854, 0.1623, 0.1562])
    ])

    print("\n📂 در حال بارگذاری دیتاست...")
    test_dataset = WildDeepfakeDataset(
        real_path=args.test_real_dir,
        fake_path=args.test_fake_dir,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    print("\n🔄 در حال بارگذاری مدل‌ها...")
    model1 = load_pruned_model(args.model1_path, device)
    model2 = load_pruned_model(args.model2_path, device)

    print("🔮 Model 1: در حال پیش‌بینی...")
    predictions1, labels = get_predictions(model1, test_loader, device)
    print("🔮 Model 2: در حال پیش‌بینی...")
    predictions2, _ = get_predictions(model2, test_loader, device)

    print("\n" + "="*70)
    print("🎯 در حال ترکیب فازی مدل‌ها...")
    print("="*70)
    final_predictions, accuracy, fusion_details = fuzzy_ensemble_binary(predictions1, predictions2, labels)

    print(f"\n✅ دقت ترکیب فازی: {accuracy * 100:.2f}%")
    print_detailed_results(labels, final_predictions, predictions1, predictions2)

    # ذخیره نتایج
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
    print("\n✅ نتایج ذخیره شدند: fuzzy_ensemble_results.pt و fuzzy_ensemble_results.csv")
    print("\n🎉 پردازش با موفقیت انجام شد!")


if __name__ == "__main__":
    main()
