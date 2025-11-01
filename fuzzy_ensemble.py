import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# Import معماری مدل pruned شما
from model.pruned_model.Resnet_final import ResNet_50_pruned_hardfakevsreal

class DeepfakeDataset(Dataset):
    """Dataset برای بارگذاری تصاویر"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(model_name='default'):
    
    transforms_dict = {
        'model1': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5212, 0.4260, 0.3811], std=[0.2486, 0.2238, 0.2211])
        ]),
        'model2': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5207, 0.4258, 0.3806], std=[0.2490, 0.2239, 0.2212])
        ]),
        'model3': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4868, 0.3972, 0.3624], std=[0.2296, 0.2066, 0.2009])
        ]),
        'model4': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4668, 0.3816, 0.3414], std=[0.2410, 0.2161, 0.2081])
        ]),
        'model5': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4923, 0.4042, 0.3624], std=[0.2446, 0.2198, 0.2141])
        ])
    }
    
    return transforms_dict.get(model_name, transforms_dict['default'])

# ==================== بخش 2: استخراج Predictions از مدل‌ها ====================
class ModelPredictor:
    
    def __init__(self, model_configs, device='cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.model_names = []
        
        print(f"🔧 Device: {self.device}")
        print(f"📦 بارگذاری {len(model_configs)} مدل ResNet-50 Pruned...")
        
        for i, config in enumerate(model_configs):
            model_path = config['model_path']
            mask_path = config.get('mask_path', None)
            name = config.get('name', f'Model_{i+1}')
            
            print(f"\n [{i+1}/{len(model_configs)}] {name}")
            print(f" مدل: {os.path.basename(model_path)}")
            if mask_path:
                print(f" Mask: {os.path.basename(mask_path)}")
            else:
                print(f" Mask: None (بدون ماسک)")
            
            model = self.load_pruned_model(model_path, mask_path)
            model.eval()
            
            self.models.append(model)
            self.model_names.append(name)
            
            print(f" ✓ بارگذاری موفق")
        
        print(f"\n✅ همه مدل‌ها با موفقیت بارگذاری شدند!")
    
    def load_pruned_model(self, model_path, mask_path=None):
        
        if mask_path:
            mask_checkpoint = torch.load(mask_path, map_location='cpu')
        
            if isinstance(mask_checkpoint, dict):
                if 'mask' in mask_checkpoint:
                    masks = mask_checkpoint['mask']
                elif 'masks' in mask_checkpoint:
                    masks = mask_checkpoint['masks']
                else:
                    # شاید خود checkpoint همان masks باشد
                    masks = mask_checkpoint
            else:
                masks = mask_checkpoint
        else:
            masks = None
        
        # ساخت مدل با masks (اگر masks None باشد، فرض بر این است که مدل بدون ماسک کار می‌کند)
        model = ResNet_50_pruned_hardfakevsreal(masks=masks)
        
        # بارگذاری وزن‌ها
        model_checkpoint = torch.load(model_path, map_location=self.device)
        
        # استخراج state_dict
        if isinstance(model_checkpoint, dict):
            if 'model_state_dict' in model_checkpoint:
                state_dict = model_checkpoint['model_state_dict']
            elif 'state_dict' in model_checkpoint:
                state_dict = model_checkpoint['state_dict']
            elif 'model' in model_checkpoint:
                state_dict = model_checkpoint['model']
            else:
                state_dict = model_checkpoint
        else:
            state_dict = model_checkpoint
        
        # بارگذاری وزن‌ها
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        
        return model
    
    def get_predictions(self, image_paths, labels, batch_size=32, show_progress=True):
        
        n_samples = len(image_paths)
        n_models = len(self.models)
        all_predictions = np.zeros((n_samples, n_models))
        
        print(f"\n🔍 استخراج پیش‌بینی‌ها از {n_models} مدل...")
        print(f" تعداد نمونه‌ها: {n_samples}")
        print(f" Batch size: {batch_size}")
        
        for model_idx, (model, model_name) in enumerate(zip(self.models, self.model_names)):
            print(f"\n 📊 مدل {model_idx+1}/{n_models}: {model_name}")
            
            # ایجاد dataset با transform مخصوص این مدل
            transform = get_transforms(model_name)
            dataset = DeepfakeDataset(image_paths, labels, transform)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            # استخراج predictions
            predictions = []
            with torch.no_grad():
                iterator = tqdm(dataloader, desc=f" Processing") if show_progress else dataloader
                for images, _ in iterator:
                    images = images.to(self.device)
                    outputs, _ = model(images) # model returns (output, feature_list)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    predictions.extend(probs)
            
            predictions = np.array(predictions)
            all_predictions[:, model_idx] = predictions
            
            # نمایش آماره‌ها
            print(f" 📈 میانگین: {np.mean(predictions):.4f}")
            print(f" 📉 انحراف معیار: {np.std(predictions):.4f}")
            print(f" 📏 دامنه: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        
        print("\n✅ استخراج پیش‌بینی‌ها تکمیل شد!")
        return all_predictions, np.array(labels)

# ==================== بخش 3: Choquet Integral ====================
class SimplifiedChoquetIntegral(nn.Module):
    
    def __init__(self, n_models=5):
        super(SimplifiedChoquetIntegral, self).__init__()
        self.n_models = n_models
        
        # وزن‌های مستقل برای هر مدل
        self.individual_weights = nn.Parameter(torch.ones(n_models))
        
        # وزن‌های تعامل زوجی (synergy)
        n_pairs = n_models * (n_models - 1) // 2
        self.interaction_weights = nn.Parameter(torch.zeros(n_pairs))
        
    def forward(self, predictions):
        """
        ترکیب فازی پیش‌بینی‌های مدل‌ها
        
        Args:
            predictions: (batch_size, n_models) - احتمالات پیش‌بینی شده
        """
        batch_size = predictions.shape[0]
        
        # نرمال‌سازی وزن‌های مستقل
        weights = torch.softmax(self.individual_weights, dim=0)
        
        # ترکیب خطی وزن‌دار
        result = torch.sum(predictions * weights.unsqueeze(0), dim=1)
        
        # اضافه کردن تعاملات زوجی
        pair_idx = 0
        for i in range(self.n_models):
            for j in range(i+1, self.n_models):
                interaction = self.interaction_weights[pair_idx]
                result += interaction * predictions[:, i] * predictions[:, j]
                pair_idx += 1
        
        # محدود کردن خروجی به [0, 1]
        result = torch.sigmoid(result)
        
        return result

class EnsembleTrainer:
    """کلاس آموزش و ارزیابی Ensemble با Choquet Integral"""
    
    def __init__(self, model_predictions, true_labels, model_names=None):
        """
        Args:
            model_predictions: آرایه numpy (n_samples, n_models)
            true_labels: آرایه numpy (n_samples,)
            model_names: لیست نام مدل‌ها (اختیاری)
        """
        self.n_models = model_predictions.shape[1]
        self.model_names = model_names or [f'Model {i+1}' for i in range(self.n_models)]
        
        # تبدیل به tensor
        self.X = torch.FloatTensor(model_predictions)
        self.y = torch.FloatTensor(true_labels)
        
        # تقسیم به train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.X.numpy(), self.y.numpy(),
            test_size=0.2, random_state=42, stratify=true_labels
        )
        
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.FloatTensor(y_val)
        
        # ساخت مدل Choquet
        self.model = SimplifiedChoquetIntegral(self.n_models)
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        
    def train(self, epochs=100, lr=0.01, batch_size=256, patience=20):
        """آموزش مدل Choquet Integral"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        n_batches = len(self.X_train) // batch_size + 1
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\n" + "="*70)
        print("🎓 شروع آموزش Choquet Integral")
        print("="*70)
        print(f"نمونه‌های آموزشی: {len(self.X_train):,}")
        print(f"نمونه‌های اعتبارسنجی: {len(self.X_val):,}")
        print(f"تعداد مدل‌ها: {self.n_models}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Patience: {patience}\n")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            indices = torch.randperm(len(self.X_train))
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.X_train))
                if start_idx >= end_idx:
                    break
                
                batch_indices = indices[start_idx:end_idx]
                batch_X = self.X_train[batch_indices]
                batch_y = self.y_train[batch_indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_val)
                val_loss = criterion(val_outputs, self.y_val).item()
                val_auc = roc_auc_score(self.y_val.numpy(), val_outputs.numpy())
            
            avg_train_loss = train_loss / n_batches
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            scheduler.step(val_loss)
            
            # نمایش پیشرفت
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️ Early stopping در epoch {epoch+1}")
                    break
        
        # بارگذاری بهترین مدل
        self.model.load_state_dict(self.best_model_state)
        print("\n✅ آموزش با موفقیت تکمیل شد!")
        print(f" بهترین Val Loss: {best_val_loss:.4f}")
        
    def evaluate(self, X_test, y_test):
        """ارزیابی جامع مدل ensemble"""
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        
        with torch.no_grad():
            ensemble_probs = self.model(X_test_tensor).numpy()
        
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        
        # متریک‌های ensemble
        ensemble_acc = accuracy_score(y_test, ensemble_preds)
        ensemble_auc = roc_auc_score(y_test, ensemble_probs)
        ensemble_f1 = f1_score(y_test, ensemble_preds)
        
        # ارزیابی مدل‌های تکی
        individual_results = []
        for i in range(X_test.shape[1]):
            preds = (X_test[:, i] >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, X_test[:, i])
            f1 = f1_score(y_test, preds)
            individual_results.append({
                'model': self.model_names[i],
                'accuracy': acc,
                'auc': auc,
                'f1': f1
            })
        
        best_single = max(individual_results, key=lambda x: x['auc'])
        
        # نمایش نتایج
        print("\n" + "="*70)
        print("📊 نتایج ارزیابی نهایی")
        print("="*70)
        
        print("\n🔍 عملکرد مدل‌های تکی:")
        print("-" * 70)
        for result in individual_results:
            print(f"{result['model']:20s} | "
                  f"Acc: {result['accuracy']:.4f} | "
                  f"AUC: {result['auc']:.4f} | "
                  f"F1: {result['f1']:.4f}")
        
        print("\n" + "-" * 70)
        print(f"🏆 بهترین مدل تکی: {best_single['model']}")
        print(f" AUC: {best_single['auc']:.4f}")
        
        print("\n🎯 عملکرد Ensemble (Choquet Integral):")
        print("-" * 70)
        print(f"Accuracy: {ensemble_acc:.4f}")
        print(f"AUC: {ensemble_auc:.4f}")
        print(f"F1-Score: {ensemble_f1:.4f}")
        
        print("\n📈 بهبود نسبت به بهترین مدل تکی:")
        print("-" * 70)
        improvement_auc = ((ensemble_auc - best_single['auc']) / best_single['auc']) * 100
        improvement_acc = ((ensemble_acc - best_single['accuracy']) / best_single['accuracy']) * 100
        improvement_f1 = ((ensemble_f1 - best_single['f1']) / best_single['f1']) * 100
        
        print(f"AUC: {improvement_auc:+.2f}%")
        print(f"Accuracy: {improvement_acc:+.2f}%")
        print(f"F1-Score: {improvement_f1:+.2f}%")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, ensemble_preds)
        print("\n📋 Confusion Matrix:")
        print(f" True Negative: {cm[0,0]:,}")
        print(f" False Positive: {cm[0,1]:,}")
        print(f" False Negative: {cm[1,0]:,}")
        print(f" True Positive: {cm[1,1]:,}")
        print("="*70)
        
        return {
            'ensemble': {'accuracy': ensemble_acc, 'auc': ensemble_auc, 'f1': ensemble_f1},
            'best_single': best_single,
            'individual': individual_results,
            'improvement': {
                'auc': improvement_auc,
                'accuracy': improvement_acc,
                'f1': improvement_f1
            }
        }
    
    def get_learned_weights(self):
        """نمایش وزن‌های یادگیری‌شده"""
        print("\n" + "="*70)
        print("⚖️ وزن‌های یادگیری‌شده")
        print("="*70)
        
        weights = torch.softmax(self.model.individual_weights, dim=0).detach().numpy()
        print("\n📊 وزن‌های مستقل مدل‌ها:")
        for i, (name, w) in enumerate(zip(self.model_names, weights)):
            bar = '█' * int(w * 50)
            print(f" {name:20s}: {w:.4f} ({w*100:5.1f}%) {bar}")
        
        print("\n🔗 وزن‌های تعامل زوجی:")
        interactions = self.model.interaction_weights.detach().numpy()
        idx = 0
        for i in range(self.n_models):
            for j in range(i+1, self.n_models):
                sign = "+" if interactions[idx] >= 0 else ""
                color = "🟢" if interactions[idx] > 0.01 else "🔴" if interactions[idx] < -0.01 else "⚪"
                print(f" {color} {self.model_names[i]} ↔ {self.model_names[j]}: {sign}{interactions[idx]:.4f}")
                idx += 1
        print("="*70)
    
    def plot_training_history(self, save_path='training_history.png'):
        """رسم تاریخچه آموزش"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2, color='#1f77b4')
        axes[0].plot(self.history['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training History - Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['val_auc'], label='Validation AUC', color='#2ca02c', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('AUC', fontsize=12)
        axes[1].set_title('Training History - AUC', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 نمودار ذخیره شد: {save_path}")
        plt.show()
    
    def save_model(self, save_path='choquet_ensemble.pth'):
        """ذخیره مدل آموزش‌دیده"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_names': self.model_names,
            'n_models': self.n_models,
            'history': self.history
        }, save_path)
        print(f"💾 مدل ذخیره شد: {save_path}")
    
    def load_model(self, load_path='choquet_ensemble.pth'):
        """بارگذاری مدل ذخیره‌شده"""
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_names = checkpoint['model_names']
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_auc': []})
        print(f"📂 مدل بارگذاری شد: {load_path}")

# ==================== بخش 4: Pipeline اصلی ====================
def main_pipeline(model_configs, image_paths, labels, test_size=0.15):
    """
    Pipeline کامل: از بارگذاری مدل‌ها تا ارزیابی نهایی
    
    Args:
        model_configs: لیست دیکشنری‌های تنظیمات مدل‌ها
        image_paths: لیست مسیرهای تصاویر
        labels: آرایه numpy برچسب‌ها
        test_size: نسبت داده‌های تست
    
    Returns:
        trainer: شیء EnsembleTrainer
        results: نتایج ارزیابی
    """
    
    print("\n" + "="*70)
    print("🚀 سیستم تشخیص Deepfake با Choquet Integral")
    print(" استفاده از مدل‌های ResNet-50 Pruned")
    print("="*70)
    
    # گام 1: بارگذاری مدل‌ها
    print("\n📥 گام 1: بارگذاری مدل‌های ResNet-50 Pruned")
    predictor = ModelPredictor(model_configs, device='cuda')
    
    # گام 2: استخراج predictions
    print("\n🔍 گام 2: استخراج پیش‌بینی‌های مدل‌ها")
    all_predictions, all_labels = predictor.get_predictions(
        image_paths, labels, batch_size=32, show_progress=True
    )
    
    # گام 3: تقسیم داده
    print("\n✂️ گام 3: تقسیم داده به Train/Test")
    X_train, X_test, y_train, y_test = train_test_split(
        all_predictions, all_labels,
        test_size=test_size, random_state=42, stratify=all_labels
    )
    print(f" آموزش: {len(X_train):,} نمونه ({(1-test_size)*100:.0f}%)")
    print(f" تست: {len(X_test):,} نمونه ({test_size*100:.0f}%)")
    
    # گام 4: آموزش Ensemble
    print("\n🎓 گام 4: آموزش مدل Ensemble")
    trainer = EnsembleTrainer(X_train, y_train, predictor.model_names)
    trainer.train(epochs=100, lr=0.01, batch_size=256, patience=20)
    
    # گام 5: ارزیابی
    print("\n📊 گام 5: ارزیابی و نمایش نتایج")
    trainer.get_learned_weights()
    results = trainer.evaluate(X_test, y_test)
    trainer.plot_training_history()
    
    # گام 6: ذخیره مدل
    print("\n💾 گام 6: ذخیره مدل نهایی")
    trainer.save_model('choquet_ensemble_final.pth')
    
    print("\n" + "="*70)
    print("✅ فرآیند با موفقیت کامل شد!")
    print("="*70)
    
    return trainer, results

# ==================== بخش 5: نحوه استفاده ====================
if __name__ == "__main__":
    import glob  # برای جمع‌آوری مسیر تصاویر
    
    # تعریف مدل‌ها (مسیرها را با مسیرهای واقعی خود جایگزین کنید)
    model_configs = [
        {
            'name': 'model1',
            'model_path': '/kaggle/input/10k-pearson-pruned/pytorch/default/1/10k_pearson_pruned.pt',
        },
        {
            'name': 'model2',
            'model_path': '/kaggle/input/140k-pearson-pruned/pytorch/default/1/140k_pearson_pruned.pt',
        },
        {
            'name': 'model3',
            'model_path': '/kaggle/input/200k-pearson-pruned/pytorch/default/1/200k_kdfs_pruned.pt',
        },
        {
            'name': 'model4',
            'model_path': '/kaggle/input/190k-pearson-pruned/pytorch/default/1/190k_pearson_pruned.pt',
        },
        {
            'name': 'model5',
            'model_path': '/kaggle/input/330k-base-pruned/pytorch/default/1/330k_base_pruned.pt',
        }
    ]
    
    # تعریف دیتاست (مسیرها را با مسیرهای واقعی خود جایگزین کنید)
    real_images = glob.glob('//kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k/valid/real/*.png')  # یا هر فرمت
    fake_images = glob.glob('//kaggle/input/20k-wild-deepfake-dataset/wild-dataset_20k/valid/fake/*.png')
    
    image_paths = real_images + fake_images
    labels = np.array([0] * len(real_images) + [1] * len(fake_images))  # ۰ برای واقعی، ۱ برای فیک
    
    # اجرای pipeline
    trainer, results = main_pipeline(
        model_configs=model_configs,
        image_paths=image_paths,
        labels=labels,
        test_size=0.15  # می‌توانید تغییر دهید
    )
    
    # اگر می‌خواهید مدل ذخیره‌شده را بعداً بارگذاری کنید:
    # trainer.load_model('choquet_ensemble_final.pth')
