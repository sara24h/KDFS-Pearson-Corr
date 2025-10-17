import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# تنظیمات GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")

# ==================== Import Model Architecture ====================
# شما باید کلاس ResNet_50_pruned_hardfakevsreal را از فایل اصلی خود import کنید
# برای مثال:
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

# اگر فایل مدل را ندارید، مسیر آن را به sys.path اضافه کنید:
import sys
# sys.path.append('/path/to/your/model/directory')

# برای Kaggle:
# اگر مدل در input قرار دارد:
#sys.path.append('/kaggle/input/your-model-code-dataset')

# سپس import کنید:
try:
    from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    print("✅ Model architecture imported successfully")
except ImportError as e:
    print(f"⚠️  Could not import ResNet_50_pruned_hardfakevsreal: {e}")
    print("Please ensure the model architecture file is available")
    ResNet_50_pruned_hardfakevsreal = None

# ==================== Dataset ====================
class DeepfakeDataset(Dataset):
    """Dataset برای بارگذاری تصاویر Deepfake"""
    def __init__(self, real_path, fake_path, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # بارگذاری تصاویر Real (label=0)
        real_images = glob.glob(os.path.join(real_path, "*.*"))
        self.images.extend(real_images)
        self.labels.extend([0] * len(real_images))
        
        # بارگذاری تصاویر Fake (label=1)
        fake_images = glob.glob(os.path.join(fake_path, "*.*"))
        self.images.extend(fake_images)
        self.labels.extend([1] * len(fake_images))
        
        print(f"Loaded {len(real_images)} real and {len(fake_images)} fake images")
    
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
            print(f"Error loading {img_path}: {e}")
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label, img_path
            return Image.new('RGB', (224, 224)), label, img_path

# ==================== Fuzzy Ensemble ====================
class FuzzyEnsemble:
    """کلاس ترکیب‌کننده فازی برای دو مدل"""
    
    @staticmethod
    def generate_rank1(score, class_no=2):
        """محاسبه رتبه با تابع Gaussian"""
        rank = np.zeros([class_no, 1])
        scores = score.reshape(-1, 1)
        for i in range(class_no):
            rank[i] = 1 - np.exp(-((scores[i] - 1) ** 2) / 2.0)
        return rank
    
    @staticmethod
    def generate_rank2(score, class_no=2):
        """محاسبه رتبه با تابع Tanh"""
        rank = np.zeros([class_no, 1])
        scores = score.reshape(-1, 1)
        for i in range(class_no):
            rank[i] = 1 - np.tanh(((scores[i] - 1) ** 2) / 2)
        return rank
    
    @staticmethod
    def fuse_two_models(res1, res2, labels, class_no=2):
        """ترکیب نتایج دو مدل با منطق فازی"""
        cnt = 0
        predictions = []
        fused_scores = []
        
        for i in range(len(res1)):
            # محاسبه رتبه‌ها برای مدل اول
            rank1_m1 = FuzzyEnsemble.generate_rank1(res1[i], class_no)
            rank2_m1 = FuzzyEnsemble.generate_rank2(res1[i], class_no)
            rank_m1 = rank1_m1 * rank2_m1
            
            # محاسبه رتبه‌ها برای مدل دوم
            rank1_m2 = FuzzyEnsemble.generate_rank1(res2[i], class_no)
            rank2_m2 = FuzzyEnsemble.generate_rank2(res2[i], class_no)
            rank_m2 = rank1_m2 * rank2_m2
            
            # جمع رتبه‌ها
            rank_sum = rank_m1 + rank_m2
            rank_sum = np.array(rank_sum)
            
            # محاسبه میانگین امتیازات
            score_sum = 1 - (res1[i] + res2[i]) / 2
            score_sum = np.array(score_sum).reshape(-1, 1)
            
            # امتیاز نهایی فازی
            fused_score = (rank_sum.T) * score_sum
            fused_scores.append(fused_score[0])
            
            # پیش‌بینی کلاس (کمترین رتبه)
            cls = np.argmin(rank_sum)
            predictions.append(cls)
            
            # شمارش پیش‌بینی‌های صحیح
            if cls < class_no and labels[i] == cls:
                cnt += 1
        
        accuracy = cnt / len(res1)
        print(f"Fuzzy Ensemble Accuracy: {accuracy:.4f}")
        
        return predictions, fused_scores, accuracy

# ==================== بارگذاری مدل Pruned ====================
def load_pruned_model_with_masks(model_path, model_class):
    """بارگذاری مدل Pruned با استفاده از masks"""
    print(f"\n{'='*70}")
    print(f"📥 Loading pruned model from: {model_path}")
    
    try:
        # بارگذاری checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # استخراج اطلاعات
        if isinstance(checkpoint, dict):
            model_state_dict = checkpoint.get('model_state_dict')
            masks = checkpoint.get('masks')
            model_arch = checkpoint.get('model_architecture', 'Unknown')
            total_params = checkpoint.get('total_params', 0)
            
            print(f"✓ Model architecture: {model_arch}")
            print(f"✓ Total parameters: {total_params:,}")
            print(f"✓ Number of masks: {len(masks) if masks else 0}")
            
            if model_state_dict is None or masks is None:
                raise ValueError("model_state_dict or masks not found in checkpoint")
            
            # ساخت مدل با استفاده از masks
            if model_class is None:
                raise ImportError("Model class not imported. Please check import statement.")
            
            model = model_class(masks=masks)
            
            # بارگذاری وزن‌ها
            model.load_state_dict(model_state_dict)
            
            print(f"✅ Model loaded successfully!")
            
            return model
        else:
            raise ValueError(f"Expected dict, got {type(checkpoint)}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def setup_model_for_inference(model):
    """آماده‌سازی مدل برای inference با DataParallel"""
    
    # استفاده از DataParallel برای چند GPU
    if torch.cuda.device_count() > 1:
        print(f"🔀 Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    
    return model

def get_predictions_binary(model, dataloader):
    """استخراج پیش‌بینی‌های مدل برای دسته‌بندی باینری"""
    all_probs = []
    all_labels = []
    all_paths = []
    
    model.eval()
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            
            # برای خروجی تک‌کلاسه (binary classification with sigmoid)
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                # خروجی: [batch_size] یا [batch_size, 1]
                outputs = outputs.squeeze()
                probs_fake = torch.sigmoid(outputs)
                probs_real = 1 - probs_fake
                probs = torch.stack([probs_real, probs_fake], dim=1)
            else:
                # خروجی: [batch_size, 2]
                probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    
    return all_probs, all_labels, all_paths

def main():
    # بررسی import مدل
    if ResNet_50_pruned_hardfakevsreal is None:
        print("\n" + "="*70)
        print("❌ CRITICAL ERROR: Model architecture not imported!")
        print("\n📋 TO FIX THIS:")
        print("1. Upload your model architecture code to Kaggle as a dataset")
        print("2. Add the dataset to your notebook")
        print("3. Update the import path in line ~30 of this code")
        print("\n   Example:")
        print("   sys.path.append('/kaggle/input/your-model-architecture-dataset')")
        print("   from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal")
        print("="*70)
        return
    
    # مسیر دیتاست
    real_path = "/kaggle/input/wild-deepfake/test/real"
    fake_path = "/kaggle/input/wild-deepfake/test/fake"
    
    # تنظیمات
    batch_size = 32
    num_workers = 4
    class_no = 2
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ساخت Dataset و DataLoader
    print("="*70)
    print("📊 Loading test dataset...")
    test_dataset = DeepfakeDataset(real_path, fake_path, transform=transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # مسیر مدل‌ها
    model1_path = "/kaggle/input/10k_finetune_wd/pytorch/default/1/10k_final_pruned_finetuned_inference_ready (1).pt"
    model2_path = "/kaggle/input/140k_finetuned_wd/pytorch/default/1/140k_final_pruned_finetuned_inference_ready (1).pt"
    
    # بارگذاری مدل 1
    print("\n" + "="*70)
    print("🔧 Loading Model 1...")
    model1 = load_pruned_model_with_masks(model1_path, ResNet_50_pruned_hardfakevsreal)
    model1 = setup_model_for_inference(model1)
    
    # بارگذاری مدل 2
    print("\n" + "="*70)
    print("🔧 Loading Model 2...")
    model2 = load_pruned_model_with_masks(model2_path, ResNet_50_pruned_hardfakevsreal)
    model2 = setup_model_for_inference(model2)
    
    # استخراج پیش‌بینی‌ها
    print("\n" + "="*70)
    print("🔮 Getting predictions from Model 1...")
    probs1, labels, paths = get_predictions_binary(model1, test_loader)
    
    print("\n🔮 Getting predictions from Model 2...")
    probs2, _, _ = get_predictions_binary(model2, test_loader)
    
    # ارزیابی مدل‌های جداگانه
    print("\n" + "="*70)
    print("📊 Model 1 Performance:")
    preds1 = np.argmax(probs1, axis=1)
    acc1 = accuracy_score(labels, preds1)
    print(f"Accuracy: {acc1:.4f}")
    print(classification_report(labels, preds1, target_names=['Real', 'Fake']))
    
    print("\n" + "="*70)
    print("📊 Model 2 Performance:")
    preds2 = np.argmax(probs2, axis=1)
    acc2 = accuracy_score(labels, preds2)
    print(f"Accuracy: {acc2:.4f}")
    print(classification_report(labels, preds2, target_names=['Real', 'Fake']))
    
    # ترکیب فازی
    print("\n" + "="*70)
    print("🔥 Fuzzy Ensemble Fusion...")
    fuzzy_ensemble = FuzzyEnsemble()
    predictions, fused_scores, ensemble_acc = fuzzy_ensemble.fuse_two_models(
        probs1, probs2, labels, class_no
    )
    
    # گزارش نهایی
    print("\n" + "="*70)
    print("🎯 Final Ensemble Performance:")
    print(classification_report(labels, predictions, target_names=['Real', 'Fake']))
    print("\n📈 Confusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    print(f"\n   Real predicted as Real: {cm[0,0]}")
    print(f"   Real predicted as Fake: {cm[0,1]}")
    print(f"   Fake predicted as Real: {cm[1,0]}")
    print(f"   Fake predicted as Fake: {cm[1,1]}")
    
    # ذخیره نتایج
    results_df = pd.DataFrame({
        'image_path': paths,
        'true_label': ['Real' if l == 0 else 'Fake' for l in labels],
        'model1_pred': ['Real' if p == 0 else 'Fake' for p in preds1],
        'model2_pred': ['Real' if p == 0 else 'Fake' for p in preds2],
        'ensemble_pred': ['Real' if p == 0 else 'Fake' for p in predictions],
        'model1_prob_real': probs1[:, 0],
        'model1_prob_fake': probs1[:, 1],
        'model2_prob_real': probs2[:, 0],
        'model2_prob_fake': probs2[:, 1],
    })
    
    results_df.to_csv('fuzzy_ensemble_results.csv', index=False)
    print("\n💾 Results saved to 'fuzzy_ensemble_results.csv'")
    
    # مقایسه عملکرد
    print("\n" + "="*70)
    print("📊 Performance Comparison:")
    print(f"   Model 1 Accuracy      : {acc1:.4f}")
    print(f"   Model 2 Accuracy      : {acc2:.4f}")
    print(f"   Fuzzy Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"   Improvement over M1   : {(ensemble_acc - acc1)*100:+.2f}%")
    print(f"   Improvement over M2   : {(ensemble_acc - acc2)*100:+.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
