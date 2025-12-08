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

# --- اضافه کردن کتابخانه‌های مورد نیاز ---
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = self.pin_memory
        self.arch = args.arch  # Expected to be 'ResNet_50'
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode  # 'hardfake', 'rvf10k', '140k', '200k', '190k', '330k'
        
        # --- اضافه کردن مسیر برای ذخیره نتایج ---
        self.result_dir = getattr(args, 'result_dir', './test_results')
        os.makedirs(self.result_dir, exist_ok=True)

        # Verify CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")

    def dataload(self):
        print("==> Loading test dataset..")
        try:
            # Verify dataset paths
            if self.dataset_mode == 'hardfake':
                csv_path = os.path.join(self.dataset_dir, 'data.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")
            elif self.dataset_mode == 'rvf10k':
                train_csv = '/kaggle/input/rvf10k/train.csv'
                valid_csv = '/kaggle/input/rvf10k/valid.csv'
                if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
                    raise FileNotFoundError(f"CSV files not found: {train_csv}, {valid_csv}")
            elif self.dataset_mode == '140k':
                test_csv = os.path.join(self.dataset_dir, 'test.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")
            elif self.dataset_mode == '200k':
                test_csv = os.path.join(self.dataset_dir, 'test_labels.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")
            elif self.dataset_mode == '190k':
                if not os.path.exists(self.dataset_dir):
                    raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
            elif self.dataset_mode == '330k':
                if not os.path.exists(self.dataset_dir):
                    raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

            # Initialize dataset based on mode
            if self.dataset_mode == 'hardfake':
                dataset = Dataset_selector(
                    dataset_mode='hardfake',
                    hardfake_csv_file=os.path.join(self.dataset_dir, 'data.csv'),
                    hardfake_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif self.dataset_mode == 'rvf10k':
                dataset = Dataset_selector(
                    dataset_mode='rvf10k',
                    rvf10k_train_csv='/kaggle/input/rvf10k/train.csv',
                    rvf10k_valid_csv='/kaggle/input/rvf10k/valid.csv',
                    rvf10k_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif self.dataset_mode == '140k':
                dataset = Dataset_selector(
                    dataset_mode='140k',
                    realfake140k_train_csv=os.path.join(self.dataset_dir, 'train.csv'),
                    realfake140k_valid_csv=os.path.join(self.dataset_dir, 'valid.csv'),
                    realfake140k_test_csv=os.path.join(self.dataset_dir, 'test.csv'),
                    realfake140k_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif self.dataset_mode == '200k':
                test_csv = os.path.join(self.dataset_dir, 'test_labels.csv')
                if not os.path.exists(test_csv):
                    raise FileNotFoundError(f"CSV file not found: {test_csv}")
                dataset = Dataset_selector(
                dataset_mode='200k',
                realfake200k_train_csv=os.path.join(self.dataset_dir, 'train_labels.csv'),
                realfake200k_val_csv=os.path.join(self.dataset_dir, 'val_labels.csv'),
                realfake200k_test_csv=os.path.join(self.dataset_dir, 'test_labels.csv'),
                realfake200k_root_dir=os.path.join(self.dataset_dir, 'my_real_vs_ai_dataset/my_real_vs_ai_dataset'), 
                train_batch_size=self.test_batch_size,
                eval_batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                ddp=False
            )
            elif self.dataset_mode == '190k':
                dataset = Dataset_selector(
                    dataset_mode='190k',
                    realfake190k_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )
            elif self.dataset_mode == '330k':
                dataset = Dataset_selector(
                    dataset_mode='330k',
                    realfake330k_root_dir=self.dataset_dir,
                    train_batch_size=self.test_batch_size,
                    eval_batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    ddp=False
                )

            self.test_loader = dataset.loader_test
            print(f"{self.dataset_mode} test dataset loaded! Total batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def build_model(self):
        print("==> Building student model..")
        try:
            print(f"Loading sparse student model for dataset mode: {self.dataset_mode}")
            self.student = ResNet_50_sparse_hardfakevsreal()
            # Load checkpoint
            if not os.path.exists(self.sparsed_student_ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
            ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt_student["student"] if "student" in ckpt_student else ckpt_student
            try:
                self.student.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"State dict loading failed with strict=True: {str(e)}")
                print("Trying with strict=False to identify mismatched keys...")
                self.student.load_state_dict(state_dict, strict=False)
                print("Loaded with strict=False; check for missing or unexpected keys.")

            self.student.to(self.device)
            print(f"Model loaded on {self.device}")
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def test(self):
        # --- متغیرهای جدید برای جمع‌آوری نتایج ---
        all_targets = []
        all_probs = []

        self.student.eval()
        self.student.ticket = True  # Enable ticket mode for sparse model
        
        print("\n==> Starting evaluation...")
        try:
            with torch.no_grad():
                with tqdm(total=len(self.test_loader), ncols=100, desc="Testing") as _tqdm:
                    for images, targets in self.test_loader:
                        images = images.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True).float()
                        
                        logits_student, _ = self.student(images)
                        logits_student = logits_student.squeeze()
                        
                        # --- جمع‌آوری احتمالات و برچسب‌ها ---
                        probs = torch.sigmoid(logits_student)
                        
                        all_targets.extend(targets.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())

                        _tqdm.update(1)
                        time.sleep(0.01)

            # --- تبدیل لیست‌ها به آرایه‌های NumPy ---
            all_targets_np = np.array(all_targets)
            all_probs_np = np.array(all_probs)
            all_preds_np = (all_probs_np > 0.5).astype(int)

            # --- محاسبه معیارها ---
            tn, fp, fn, tp = confusion_matrix(all_targets_np, all_preds_np).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            fpr, tpr, _ = roc_curve(all_targets_np, all_probs_np)
            roc_auc = auc(fpr, tpr)

            # --- چاپ نتایج در کنسول ---
            print("\n" + "="*50)
            print("           FINAL TEST RESULTS")
            print("="*50)
            print(f"Dataset: {self.dataset_mode}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1_score:.4f}")
            print(f"AUC: {roc_auc:.4f}")
            print("-"*50)
            print(f"True Positives (TP): {tp}")
            print(f"False Positives (FP): {fp}")
            print(f"True Negatives (TN): {tn}")
            print(f"False Negatives (FN): {fn}")
            print("="*50 + "\n")

            # --- ذخیره نتایج در فایل متنی ---
            report_path = os.path.join(self.result_dir, 'test_report.txt')
            with open(report_path, 'w') as f:
                f.write("="*50 + "\n")
                f.write("           FINAL TEST RESULTS\n")
                f.write("="*50 + "\n")
                f.write(f"Dataset: {self.dataset_mode}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1_score:.4f}\n")
                f.write(f"AUC: {roc_auc:.4f}\n")
                f.write("-"*50 + "\n")
                f.write(f"True Positives (TP): {tp}\n")
                f.write(f"False Positives (FP): {fp}\n")
                f.write(f"True Negatives (TN): {tn}\n")
                f.write(f"False Negatives (FN): {fn}\n")
                f.write("="*50 + "\n")
                f.write("\nClassification Report:\n")
                f.write(classification_report(all_targets_np, all_preds_np, target_names=['Real', 'Fake']))
            print(f"Test report saved to: {report_path}")

            # --- رسم و ذخیره نمودار ROC ---
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            roc_path = os.path.join(self.result_dir, 'roc_curve.png')
            plt.savefig(roc_path)
            plt.close()
            print(f"ROC curve saved to: {roc_path}")

            # --- رسم و ذخیره ماتریس درهم‌ریختگی ---
            cm = confusion_matrix(all_targets_np, all_preds_np)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Predicted Real', 'Predicted Fake'], 
                        yticklabels=['Actual Real', 'Actual Fake'])
            plt.title('Confusion Matrix')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            cm_path = os.path.join(self.result_dir, 'confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            print(f"Confusion Matrix saved to: {cm_path}")

            # Calculate FLOPs and parameters
            (
                Flops_baseline,
                Flops,
                Flops_reduction,
                Params_baseline,
                Params,
                Params_reduction,
                
            ) = get_flops_and_params(self.dataset_mode, self.sparsed_student_ckpt_path)
            print(
                f"\nParams_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, "
                f"Params reduction: {Params_reduction:.2f}%"
            )
            print(
                f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, "
                f"Flops reduction: {Flops_reduction:.2f}%"
            )
           
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            raise

    def main(self):
        print(f"Starting test pipeline with dataset mode: {self.dataset_mode}")
        print(f"Results will be saved in: {self.result_dir}")
        try:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"Device name: {torch.cuda.get_device_name(0)}")

            self.dataload()
            self.build_model()
            self.test()
        except Exception as e:
            print(f"Error in test pipeline: {str(e)}")
            raise
