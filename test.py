import os
import time
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from data.dataset import Dataset_selector
from get_flops_and_params import get_flops_and_params


class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = getattr(args, 'num_workers', 4)
        self.pin_memory = getattr(args, 'pin_memory', True)
        self.arch = getattr(args, 'arch', 'resnet50')
        self.device = args.device
        self.test_batch_size = getattr(args, 'test_batch_size', 256)
        self.ckpt_paths = args.ckpt_paths
        self.model_names = getattr(args, 'model_names', [f"Model_{i}" for i in range(len(self.ckpt_paths))])
        self.dataset_mode = args.dataset_mode
        self.result_dir = getattr(args, 'result_dir', './test_results')
        os.makedirs(self.result_dir, exist_ok=True)

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")
        if len(self.ckpt_paths) != len(self.model_names):
            raise ValueError("ckpt_paths and model_names must have same length.")

    def dataload(self):
        print("==> Loading test dataset..")
        try:
            mode = self.dataset_mode
            ddir = self.dataset_dir

            if mode == 'rvf10k':
                train_csv = '/kaggle/input/rvf10k/train.csv'
                valid_csv = '/kaggle/input/rvf10k/valid.csv'
                if not (os.path.exists(train_csv) and os.path.exists(valid_csv)):
                    raise FileNotFoundError("RVF10k CSVs not found")
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
                if not os.path.exists(test_csv): raise FileNotFoundError(test_csv)
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
                if not os.path.exists(ddir): raise FileNotFoundError(ddir)
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
            print(f"{mode} test dataset loaded! Batches: {len(self.test_loader)}")
        except Exception as e:
            print(f"Dataset load error: {e}")
            raise

    def build_model(self, ckpt_path):
        print(f"==> Loading model from: {ckpt_path}")
        model = ResNet_50_sparse_hardfakevsreal()
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state_dict = ckpt.get('student', ckpt)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict load failed: {e}. Trying strict=False.")
            model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        model.ticket = True
        return model

    def evaluate_model(self, model):
        all_targets, all_probs = [], []
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Inference", ncols=100):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float()
                logits, _ = model(images)
                probs = torch.sigmoid(logits.squeeze(-1) if logits.dim() > 1 else logits)
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_targets), np.array(all_probs)

    def test(self):
        plt.figure(figsize=(8, 6))
        all_targets_ref = None

        for ckpt, name in zip(self.ckpt_paths, self.model_names):
            model = self.build_model(ckpt)
            targets, probs = self.evaluate_model(model)

            if all_targets_ref is None:
                all_targets_ref = targets
            else:
                assert np.array_equal(all_targets_ref, targets), "GT mismatch between models!"

            fpr, tpr, _ = roc_curve(targets, probs)
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
            print(f"[{name}] AUC = {auc_val:.4f}")

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        roc_path = os.path.join(self.result_dir, 'roc_curves_comparison.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✅ ROC comparison saved to: {roc_path}")

        # Full report for first model
        model = self.build_model(self.ckpt_paths[0])
        targets, probs = self.evaluate_model(model)
        preds = (probs > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        _, _, auc_val = roc_curve(targets, probs), None, auc(*roc_curve(targets, probs)[:2])

        # Save report
        with open(os.path.join(self.result_dir, 'test_report.txt'), 'w') as f:
            f.write(f"Dataset: {self.dataset_mode}\n")
            f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\nAUC: {auc_val:.4f}\n")
            f.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
            f.write("\n" + classification_report(targets, preds, target_names=['Real', 'Fake']))

        # Confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(targets, preds), annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix (Model 1)')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.result_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # FLOPs & Params
        Flops_b, Flops, Flops_red, Params_b, Params, Params_red = get_flops_and_params(
            self.dataset_mode, self.ckpt_paths[0]
        )
        print(f"\nParams: {Params_b:.2f}M → {Params:.2f}M (↓{Params_red:.1f}%)")
        print(f"FLOPs:  {Flops_b:.2f}M → {Flops:.2f}M (↓{Flops_red:.1f}%)")

    def main(self):
        print(f"🚀 Testing {len(self.ckpt_paths)} models on '{self.dataset_mode}'")
        print(f"💾 Results dir: {self.result_dir}")
        self.dataload()
        self.test()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k', '200k', '190k', '330k'])
    parser.add_argument('--ckpt1', type=str, required=True, help='Path to first model checkpoint')
    parser.add_argument('--ckpt2', type=str, required=True, help='Path to second model checkpoint')
    parser.add_argument('--name1', type=str, default='Model A')
    parser.add_argument('--name2', type=str, default='Model B')
    parser.add_argument('--result_dir', type=str, default='./test_results')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    # Build args object compatible with Test class
    class Args:
        dataset_dir = args.dataset_dir
        dataset_mode = args.dataset_mode
        ckpt_paths = ['/kaggle/input/10k-kdfs-seed-2025-data/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt', '/kaggle/input/10k-pearson-seed5555-data/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt']
        model_names = [args.name1, args.name2]
        result_dir = args.result_dir
        test_batch_size = args.batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tester = Test(Args())
    tester.main()
