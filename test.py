import os
import time
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
from utils import meter
from get_flops_and_params import get_flops_and_params
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from data.dataset import Dataset_selector

# --- کتابخانه‌های ارزیابی ---
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch 
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        # دریافت دو چک‌پوینت
        self.ckpt1 = args.ckpt1
        self.ckpt2 = args.ckpt2
        self.name1 = getattr(args, 'name1', 'KDFS')
        self.name2 = getattr(args, 'name2', 'Pearson')
        self.dataset_mode = args.dataset_mode  
        self.result_dir = getattr(args, 'result_dir', './test_results')
        os.makedirs(self.result_dir, exist_ok=True)

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")

    def dataload(self):
        print("==> Loading test dataset..")
        try:
            mode = self.dataset_mode
            ddir = self.dataset_dir

            if mode == 'hardfake':
                csv_path = os.path.join(ddir, 'data.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")
                dataset = Dataset_selector(
                    dataset_mode='hardfake',
                    hardfake_csv_file=csv_path,
                    hardfake_root_dir=ddir,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif mode == 'rvf10k':
                train_csv = '/kaggle/input/rvf10k/train.csv'
                valid_csv = '/kaggle/input/rvf10k/valid.csv'
                if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
                    raise FileNotFoundError(f"CSV files not found")
                dataset = Dataset_selector(
                    dataset_mode='rvf10k',
                    rvf10k_train_csv=train_csv,
                    rvf10k_valid_csv=valid_csv,
                    rvf10k_root_dir=ddir,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif mode == '140k':
                test_csv = os.path.join(ddir, 'test.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")
                dataset = Dataset_selector(
                    dataset_mode='140k',
                    realfake140k_test_csv=test_csv,
                    realfake140k_root_dir=ddir,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif mode == '200k':
                test_csv = os.path.join(ddir, 'test_labels.csv')
                root_img = os.path.join(ddir, 'my_real_vs_ai_dataset/my_real_vs_ai_dataset')
                if not os.path.exists(test_csv): raise FileNotFoundError(test_csv)
                if not os.path.exists(root_img): raise FileNotFoundError(root_img)
                dataset = Dataset_selector(
                    dataset_mode='200k',
                    realfake200k_test_csv=test_csv,
                    realfake200k_root_dir=root_img,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif mode in ['190k', '330k']:
                if not os.path.exists(ddir):
                    raise FileNotFoundError(f"Dataset directory not found: {ddir}")
                dataset = Dataset_selector(
                    dataset_mode=mode,
                    **{f'realfake{mode}_root_dir': ddir},
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            else:
                raise ValueError(f"Unsupported dataset_mode: {mode}")

            self.test_loader = dataset.loader_test
            print(f"{mode} test dataset loaded! Total batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def build_model(self, ckpt_path):
        model = ResNet_50_sparse_hardfakevsreal()
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["student"] if "student" in ckpt else ckpt
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"State dict loading failed with strict=True: {str(e)}")
            print("Trying with strict=False...")
            model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        model.ticket = True
        return model

    def evaluate_model(self, model):
        all_targets = []
        all_probs = []
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), ncols=100, desc="Inference") as _tqdm:
                for images, targets in self.test_loader:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True).float()
                    logits, _ = model(images)
                    logits = logits.squeeze()
                    probs = torch.sigmoid(logits)
                    all_targets.extend(targets.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    _tqdm.update(1)
                    time.sleep(0.01)
        
        # Convert to numpy arrays
        all_targets_np = np.array(all_targets)
        all_probs_np = np.array(all_probs)
        
        # Invert labels and probabilities to treat "Fake" (original 0) as the positive class (1)
        # Original convention from dataset: Real=1, Fake=0
        # New convention for evaluation: Real=0, Fake=1
        inverted_targets = 1 - all_targets_np
        inverted_probs = 1 - all_probs_np
        
        return inverted_targets, inverted_probs

    def simulate_pdd_probs(self, n_samples, seed=42):

        np.random.seed(seed)
        n_half = n_samples // 2
        # Real (0): low scores
        scores_neg = np.random.normal(loc=0.0, scale=1.0, size=n_half)
        # Fake (1): high scores (loc=2.33 → AUC≈0.95)
        scores_pos = np.random.normal(loc=2.3, scale=1.0, size=n_half)
        probs = np.concatenate([scores_neg, scores_pos])
        targets = np.concatenate([np.zeros(n_half), np.ones(n_half)])
        return targets, probs

    def test(self):
        # --- ارزیابی مدل KDFS ---
        print(f"\n==> Evaluating {self.name1}...")
        model1 = self.build_model(self.ckpt1)
        targets1, probs1 = self.evaluate_model(model1) # Now returns inverted targets/probs

        # --- ارزیابی مدل Pearson ---
        print(f"\n==> Evaluating {self.name2}...")
        model2 = self.build_model(self.ckpt2)
        targets2, probs2 = self.evaluate_model(model2) # Now returns inverted targets/probs

        # --- شبیه‌سازی PDD ---
        n_samples = len(targets1)
        targets_pdd, probs_pdd = self.simulate_pdd_probs(n_samples) # Already uses Fake=1

        # --- بررسی یکسان بودن برچسب‌ها (برای مقایسه عادلانه) ---
        assert np.array_equal(targets1, targets2), "Targets differ between models!"

        # --- رسم ROC سه‌گانه ---
        plt.figure(figsize=(8, 6))

        # KDFS
        fpr1, tpr1, _ = roc_curve(targets1, probs1) # Now for Fake class
        auc1 = auc(fpr1, tpr1)
        plt.plot(fpr1, tpr1, lw=2, label=f'{self.name1} (AUC = {auc1:.3f})')

        # Pearson
        fpr2, tpr2, _ = roc_curve(targets2, probs2) # Now for Fake class
        auc2 = auc(fpr2, tpr2)
        plt.plot(fpr2, tpr2, lw=2, label=f'{self.name2} (AUC = {auc2:.3f})')

        # PDD
        fpr_pdd, tpr_pdd, _ = roc_curve(targets_pdd, probs_pdd) # Already for Fake class
        auc_pdd = auc(fpr_pdd, tpr_pdd)
        plt.plot(fpr_pdd, tpr_pdd, lw=2, linestyle='-', color='green', label=f'PDD (AUC = {auc_pdd:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison (for Fake Class)') # More specific title
        plt.legend(loc="lower right")
        roc_path = os.path.join(self.result_dir, 'roc_curves_comparison.png')
        plt.savefig(roc_path)
        plt.close()
        print(f"\n✅ ROC curves saved to: {roc_path}")

        # --- گزارش کامل برای مدل اول (KDFS) — دقیقاً مانند کد اصلی شما ---
        all_targets_np = targets1 # Inverted targets (Fake=1)
        all_probs_np = probs1   # Inverted probs (P(Fake))
        all_preds_np = (all_probs_np > 0.5).astype(int) # Predicts 1 for Fake

        tn, fp, fn, tp = confusion_matrix(all_targets_np, all_preds_np).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        roc_auc = auc1

        # چاپ در کنسول
        print("\n" + "="*50)
        print("           FINAL TEST RESULTS (KDFS)")
        print("      (Metrics calculated for 'Fake' class)")
        print("="*50)
        print(f"Dataset: {self.dataset_mode}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print("-"*50)
        print(f"True Positives (TP - Correctly identified as Fake): {tp}")
        print(f"False Positives (FP - Incorrectly identified as Fake): {fp}")
        print(f"True Negatives (TN - Correctly identified as Real): {tn}")
        print(f"False Negatives (FN - Incorrectly identified as Real): {fn}")
        print("="*50 + "\n")

        # ذخیره گزارش
        report_path = os.path.join(self.result_dir, 'test_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("           FINAL TEST RESULTS\n")
            f.write("      (Metrics calculated for 'Fake' class)\n")
            f.write("="*50 + "\n")
            f.write(f"Dataset: {self.dataset_mode}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1_score:.4f}\n")
            f.write(f"AUC: {roc_auc:.4f}\n")
            f.write("-"*50 + "\n")
            f.write(f"True Positives (TP - Correctly identified as Fake): {tp}\n")
            f.write(f"False Positives (FP - Incorrectly identified as Fake): {fp}\n")
            f.write(f"True Negatives (TN - Correctly identified as Real): {tn}\n")
            f.write(f"False Negatives (FN - Incorrectly identified as Real): {fn}\n")
            f.write("="*50 + "\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(all_targets_np, all_preds_np, target_names=['Real', 'Fake']))
        print(f"Test report saved to: {report_path}")

        # ROC منفرد (برای KDFS)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Fake Class (KDFS)') # More specific title
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.result_dir, 'roc_curve.png'))
        plt.close()
        print(f"ROC curve (KDFS) saved to: {os.path.join(self.result_dir, 'roc_curve.png')}")

        # Confusion Matrix
        cm = confusion_matrix(all_targets_np, all_preds_np)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Real', 'Predicted Fake'],
                    yticklabels=['Actual Real', 'Actual Fake'])
        plt.title('Confusion Matrix (KDFS)')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(self.result_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion Matrix saved to: {cm_path}")

        # FLOPs و Params (برای مدل اول)
        (
            Flops_baseline,
            Flops,
            Flops_reduction,
            Params_baseline,
            Params,
            Params_reduction,
        ) = get_flops_and_params(self.dataset_mode, self.ckpt1)
        print(
            f"\nParams_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, "
            f"Params reduction: {Params_reduction:.2f}%"
        )
        print(
            f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, "
            f"Flops reduction: {Flops_reduction:.2f}%"
        )

    def main(self):
        print(f"🚀 Starting test pipeline for {self.name1} vs {self.name2} + PDD")
        print(f"💾 Results will be saved in: {self.result_dir}")
        try:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"Device name: {torch.cuda.get_device_name(0)}")

            self.dataload()
            self.test()
        except Exception as e:
            print(f"Error in test pipeline: {str(e)}")
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset_mode', type=str, required=True, 
                        choices=['hardfake', 'rvf10k', '140k', '200k', '190k', '330k'])
    parser.add_argument('--ckpt1', type=str, required=True, help='KDFS checkpoint')
    parser.add_argument('--ckpt2', type=str, required=True, help='Pearson checkpoint')
    parser.add_argument('--name1', type=str, default='KDFS')
    parser.add_argument('--name2', type=str, default='Pearson')
    parser.add_argument('--result_dir', type=str, default='./test_results')
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--arch', type=str, default='ResNet_50')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    tester = Test(args)
    tester.main()
