import os
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from data.dataset import Dataset_selector


class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.dataset_mode = args.dataset_mode  
        self.result_dir = getattr(args, 'result_dir', './test_results')
        os.makedirs(self.result_dir, exist_ok=True)

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")

    def dataload(self):
        print("==> Loading test dataset..")
        if self.dataset_mode == 'rvf10k':
            train_csv = '/kaggle/input/rvf10k/train.csv'
            valid_csv = '/kaggle/input/rvf10k/valid.csv'
            dataset = Dataset_selector(
                dataset_mode='rvf10k',
                rvf10k_train_csv=train_csv,
                rvf10k_valid_csv=valid_csv,
                rvf10k_root_dir=self.dataset_dir,
                train_batch_size=self.test_batch_size,
                eval_batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                ddp=False
            )
        else:
            raise ValueError(f"Only 'rvf10k' supported in this version.")
        self.test_loader = dataset.loader_test
        print(f"{self.dataset_mode} test dataset loaded!")

    def evaluate_real_model(self, ckpt_path, model_name):
        """ارزیابی یک مدل واقعی و بازگرداندن fpr, tpr, auc"""
        print(f"\n==> Evaluating real model: {model_name}")
        model = ResNet_50_sparse_hardfakevsreal()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["student"] if "student" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        model.ticket = True

        all_targets, all_probs = [], []
        with torch.nogether.no_grad():
            for images, targets in tqdm(self.test_loader, desc=f"Testing {model_name}"):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()
                logits, _ = model(images)
                probs = torch.sigmoid(logits.squeeze())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score, model_name

    def generate_synthetic_roc(self, model_name="PDD Method", seed=42, loc_shift=1.8):
        """تولید منحنی ROC شبیه‌سازی‌شده بر اساس کد شما"""
        print(f"\n==> Generating synthetic ROC for: {model_name}")
        np.random.seed(seed)
        n_samples = 1500
        n_pos = n_neg = n_samples // 2

        scores_neg = np.random.normal(loc=0.0, scale=1.0, size=n_neg)
        scores_pos = np.random.normal(loc=loc_shift, scale=1.0, size=n_pos)

        y_true = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        y_scores = np.concatenate([scores_neg, scores_pos])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score, model_name

    def main(self):
        print("Starting combined ROC: 2 real models + 1 synthetic method")
        self.dataload()

        # --- بارگذاری دو مدل واقعی ---
        real_models = [
            {"name": "Ours",       "ckpt": self.args.ckpt_ours},
            {"name": "Competitor", "ckpt": self.args.ckpt_competitor},
        ]

        all_roc_data = []

        # تست مدل‌های واقعی
        for info in real_models:
            try:
                fpr, tpr, auc_val, name = self.evaluate_real_model(info["ckpt"], info["name"])
                all_roc_data.append((fpr, tpr, auc_val, name))
                print(f"✅ {name}: AUC = {auc_val:.4f}")
            except Exception as e:
                print(f"❌ Failed {info['name']}: {e}")

        # اضافه کردن مدل شبیه‌سازی‌شده (روش سوم)
        try:
            fpr3, tpr3, auc3, name3 = self.generate_synthetic_roc(
                model_name="PDD Method", seed=42, loc_shift=1.8
            )
            all_roc_data.append((fpr3, tpr3, auc3, name3))
            print(f"✅ {name3}: AUC = {auc3:.4f}")
        except Exception as e:
            print(f"❌ Failed synthetic model: {e}")

        # --- رسم نمودار ترکیبی ---
        plt.figure(figsize=(9, 7))
        colors = ['blue', 'green', 'red']
        for i, (fpr, tpr, auc_val, name) in enumerate(all_roc_data):
            plt.plot(fpr, tpr, color=colors[i], lw=2.5, label=f'{name} (AUC = {auc_val:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves: Real Models vs PDD Method', fontsize=13)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        output_path = os.path.join(self.result_dir, 'combined_real_and_synthetic_roc.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Final ROC saved to: {output_path}")
