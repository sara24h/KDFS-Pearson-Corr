import os
import random
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # برای ذخیره CSV

# فایل‌های پروژه شما
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from utils import meter

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed set to {seed} for reproducibility.")


class Test:
    def __init__(self, config):
        self.config = config
        set_seed(self.config.seed)

        self.dataset_dir = self.config.dataset_dir
        self.num_workers = self.config.num_workers
        self.pin_memory = self.config.pin_memory
        self.device = self.config.device
        self.test_batch_size = self.config.test_batch_size
        self.sparsed_student_ckpt_path = self.config.sparsed_student_ckpt_path
        self.dataset_mode = self.config.dataset_mode
        self.result_dir = self.config.result_dir
        self.new_dataset_dir = self.config.new_dataset_dir
        self.model_name = self.config.model_name  # مثلاً "Model1"

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.new_test_loader = None
        self.student = None

    def dataload(self):
        print("==> Loading datasets...")
        image_size = (256, 256)
        mean_190k = [0.4668, 0.3816, 0.3414]
        std_190k = [0.2410, 0.2161, 0.2081]

        transform_val_test_190k = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_190k, std=std_190k),
        ])

        params = {
            'dataset_mode': self.dataset_mode,
            'eval_batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'ddp': False
        }

        if self.dataset_mode == '190k':
            params['realfake190k_root_dir'] = self.dataset_dir
        # می‌تونی بقیه حالت‌ها رو هم اضافه کنی اگر لازم شد

        dataset_manager = Dataset_selector(**params)
        dataset_manager.loader_test.dataset.transform = transform_val_test_190k
        self.test_loader = dataset_manager.loader_test

        if self.new_dataset_dir:
            new_params = {
                'dataset_mode': 'new_test',
                'eval_batch_size': self.test_batch_size,
                'num_workers': self.num_workers,
                'pin_memory': self.pin_memory,
                'new_test_csv': os.path.join(self.new_dataset_dir, 'test.csv'),
                'new_test_root_dir': self.new_dataset_dir
            }
            new_dataset_manager = Dataset_selector(**new_params)
            new_dataset_manager.loader_test.dataset.transform = transform_val_test_190k
            self.new_test_loader = new_dataset_manager.loader_test

        print(f"Test loader ready with {len(self.test_loader.dataset)} samples.")

    def build_model(self):
        print("==> Building and loading model...")
        self.student = ResNet_50_sparse_hardfakevsreal()

        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.sparsed_student_ckpt_path}")

        ckpt = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        state_dict = ckpt.get("student", ckpt)
        self.student.load_state_dict(state_dict, strict=False)

        self.student.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            self.student.fc
        )

        self.student.to(self.device)
        self.student.eval()
        print(f"Model '{self.model_name}' loaded on {self.device}")

    def compute_metrics(self, loader, description="Test", save_for_mcnemar=False):
        all_preds = []
        all_targets = []
        sample_info = []

        self.student.eval()
        self.student.ticket = True

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=description, ncols=100)):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()

                logits, _ = self.student(images)
                logits = logits.squeeze()
                preds = (torch.sigmoid(logits) > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                batch_size = images.size(0)
                start_idx = batch_idx * loader.batch_size
                for i in range(batch_size):
                    try:
                        img_path = loader.dataset.samples[start_idx + i][0]
                    except:
                        img_path = f"sample_{start_idx + i}"

                    true_label = int(targets[i].item())
                    pred_label = int(preds[i].item())
                    is_correct = (pred_label == true_label)

                    sample_info.append({
                        'id': os.path.basename(img_path),
                        'true_label': true_label,        # 0 = Real, 1 = Fake
                        'pred_label': pred_label,
                        'is_correct': int(is_correct)    # 1 = درست، 0 = غلط
                    })

        # ذخیره فقط اگر بخواهیم برای مک‌نمار استفاده کنیم
        if save_for_mcnemar:
            os.makedirs(self.result_dir, exist_ok=True)
            csv_filename = f"mcnemar_ready_{self.model_name}_{self.dataset_mode}.csv"
            csv_path = os.path.join(self.result_dir, csv_filename)

            df = pd.DataFrame(sample_info)
            df.to_csv(csv_path, index=False)
            print(f"\nاطلاعات مورد نیاز آزمون مک‌نمار ذخیره شد:")
            print(f"   فایل: {csv_path}")
            print(f"   تعداد نمونه: {len(df)} نمونه")
            print(f"   ستون‌ها: id, true_label, pred_label, is_correct")
            print(f"   آماده برای ترکیب با مدل دیگر و اجرای آزمون مک‌نمار")

        # محاسبه متریک‌ها (اختیاری نمایش)
        accuracy = np.mean([s['is_correct'] for s in sample_info]) * 100
        print(f"\n[{description}] Accuracy: {accuracy:.2f}%")

        return {'sample_info': sample_info}

    def finetune(self):
        # این بخش فقط برای فاین‌تیون کردن هست و تغییری نمی‌خواد
        # (کد فاین‌تیونینگ قبلی شما بدون تغییر کپی شد)
        print("==> Fine-tuning (layer4 + fc)...")
        for name, param in self.student.named_parameters():
            param.requires_grad = 'fc' in name or 'layer4' in name

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=self.config.f_lr,
            weight_decay=self.config.weight_decay
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        best_acc = 0.0
        best_path = os.path.join(self.result_dir, f"best_finetuned_{self.model_name}.pth")

        for epoch in range(self.config.f_epochs):
            self.student.train()
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.f_epochs}"):
                images, targets = images.to(self.device), targets.to(self.device).float()
                optimizer.zero_grad()
                logits, _ = self.student(images)
                loss = criterion(logits.squeeze(), targets)
                loss.backward()
                optimizer.step()

            # اعتبارسنجی سریع
            val_acc = self.compute_metrics(self.val_loader, "Val", save_for_mcnemar=False)['sample_info']
            val_acc = np.mean([s['is_correct'] for s in val_acc]) * 100
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.student.state_dict(), best_path)

        # بارگذاری بهترین مدل
        if os.path.exists(best_path):
            self.student.load_state_dict(torch.load(best_path))
            print(f"Best finetuned model loaded (Val Acc: {best_acc:.2f}%)")

    def main(self):
        print(f"\nشروع تست مدل: {self.config.model_name}")
        self.dataload()
        self.build_model()

        print("\n--- تست قبل از فاین‌تیونینگ ---")
        self.compute_metrics(self.test_loader, "Before Finetune")

        print("\n--- شروع فاین‌تیونینگ ---")
        self.finetune()

        print("\n--- تست نهایی (بعد از فاین‌تیونینگ) + ذخیره برای مک‌نمار ---")
        self.compute_metrics(
            self.test_loader,
            description="After Finetune",
            save_for_mcnemar=True  # فقط این یکی ذخیره می‌شه
        )

        if self.new_test_loader:
            print("\n--- تست روی دیتاست جدید ---")
            self.compute_metrics(self.new_test_loader, "New Dataset Test")


class Config:
    def __init__(self):
        self.seed = 3407
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_mode = "rvf10k"
        self.dataset_dir = "/kaggle/input/rvf10k"      # تغییر بده
        self.sparsed_student_ckpt_path = "/kaggle/input/190k-pearson-seed2025-data/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt"  # تغییر بده
        self.result_dir = "kaggle/working/"
        self.model_name = "Model1" 
        self.new_dataset_dir = None
        self.test_batch_size = 64
        self.num_workers = 4
        self.pin_memory = True
        self.f_epochs = 10
        self.f_lr = 0.001
        self.weight_decay = 0.0001


if __name__ == "__main__":
    config = Config()
    tester = Test(config)
    tester.main()
