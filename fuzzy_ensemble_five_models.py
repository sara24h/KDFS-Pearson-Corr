import argparse
from model.pruned_model.Resnet_final import ResNet_50_pruned_hardfakevsreal
import torch
import torch.nn as nn
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
                self.labels.append(1)  # 1 for real
        else:
            raise FileNotFoundError(f"real folder not found: {real_path}")
        
        if os.path.exists(fake_path):
            fake_files = [f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in fake_files:
                self.images.append(os.path.join(fake_path, fname))
                self.labels.append(0)  # 0 for fake
        else:
            raise FileNotFoundError(f"fake folder not found: {fake_path}")
        
        print(f"number of Real images: {len([l for l in self.labels if l==1])}")
        print(f"number of Fake images: {len([l for l in self.labels if l==0])}")
        print(f"sum of images: {len(self.images)}")
    
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
            print(f"❌ error in loading {img_path}: {e}")
            return torch.zeros(3, 256, 256), label, img_path


def load_pruned_model(checkpoint_path, device):
    print(f"loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        masks = checkpoint.get('masks', None)
        if masks is not None:
            masks_detached = [m.detach().clone() if m is not None else None for m in masks]
        else:
            masks_detached = None
            print("masks not found")
        
        model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("weights loaded from 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("weights loaded from 'state_dict'")
        else:
            model.load_state_dict(checkpoint)
            print("weights directly loaded from checkpoint")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"number of all parameters: {total_params:,}")
    else:
        raise ValueError("checkpoint's format is not valid")
    
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
            probs_real = torch.sigmoid(outputs).squeeze()
            probs_fake = 1 - probs_real
            probs_2class = torch.stack([probs_fake, probs_real], dim=1)  # 0: fake, 1: real
            all_probs.append(probs_2class.cpu().numpy())
            all_labels.append(labels.numpy())
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    return all_probs, all_labels

# === تابع جدید برای میانگین‌گیری وزنی ===
def simple_weighted_ensemble(model_probs_list, labels, weights=None):
    """
    Performs simple or weighted averaging of model probabilities.
    model_probs_list: list of numpy arrays, each of shape (N_samples, class_no).
    labels: true labels for the samples.
    weights: optional list of weights for each model. If None, simple averaging is used.
    """
    num_models = len(model_probs_list)
    if weights is None:
        # Simple averaging
        weights = np.ones(num_models) / num_models
        method_name = "Simple Averaging"
    else:
        # Weighted averaging
        weights = np.array(weights)
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        method_name = "Weighted Averaging"

    # Stack probabilities into a single array of shape (num_models, N_samples, class_no)
    stacked_probs = np.stack(model_probs_list, axis=0)
    
    # Perform weighted average across the model axis (axis=0)
    # Reshape weights to be broadcastable: (num_models, 1, 1)
    reshaped_weights = weights.reshape(num_models, 1, 1)
    weighted_probs = np.sum(stacked_probs * reshaped_weights, axis=0)
    
    # Get final predictions by taking the argmax
    final_predictions = np.argmax(weighted_probs, axis=1)
    
    # Calculate accuracy
    accuracy = (final_predictions == labels).mean()
    
    return final_predictions, accuracy, method_name

def fuzzy_ensemble_multi(model_probs_list, labels, class_no=2):
    """
    Perform fuzzy rank-based ensemble using Gompertz function as described in the paper.
    model_probs_list: list of numpy arrays, each of shape (N_samples, class_no), with probs summing to 1 per sample.
    Assumes class 0: fake (label 0), class 1: real (label 1)
    """
    num_models = len(model_probs_list)
    num_samples = len(labels)
    correct = 0
    predictions = []
    fusion_details = []

    for i in range(num_samples):
        frs = np.zeros(class_no)
        ccfs = np.zeros(class_no)
        for c in range(class_no):
            rank_sum = 0.0
            cf_sum = 0.0
            for m in range(num_models):
                cf = model_probs_list[m][i, c]
                # Compute fuzzy rank using re-parameterized Gompertz function
                r = 1 - np.exp(-np.exp(-2.0 * cf))
                # Determine if this class is the top predicted class for this model (k=1 for binary)
                top_class = np.argmax(model_probs_list[m][i])
                if c == top_class:
                    rank_sum += r
                    cf_sum += cf
                else:
                    rank_sum += 0.5  # Penalty for rank
                    cf_sum += 0.0     # Penalty for confidence
            frs[c] = rank_sum
            ccfs[c] = cf_sum / num_models  # Average confidence (with penalties)
        
        # Final decision score
        fds = frs * ccfs
        cls = np.argmax(fds)  # Highest FDS is the best class (as per paper's "highest combined score")
        
        predictions.append(cls)
        
        fusion_details.append({
            'sample_idx': i,
            'frs': frs.flatten(),
            'ccfs': ccfs.flatten(),
            'fds': fds.flatten(),
            'prediction': cls,
            'true_label': labels[i]
        })
        
        if cls == labels[i]:
            correct += 1

    accuracy = correct / num_samples
    return np.array(predictions), accuracy, "Fuzzy Ensemble (Gompertz)", fusion_details


def print_comparison_report(labels, results_list):
    """
    Prints a final comparison report for all ensemble methods.
    results_list: A list of dictionaries, each containing results for a method.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "="*80)
    print("🏆 FINAL COMPARISON REPORT")
    print("="*80)

    # Sort results by accuracy for better presentation
    results_list.sort(key=lambda x: x['accuracy'], reverse=True)

    for i, res in enumerate(results_list):
        print(f"\n--- Method {i+1}: {res['name']} ---")
        print(f"Accuracy: {res['accuracy']*100:.4f}%")
        print("Classification Report:")
        print(classification_report(labels, res['predictions'], target_names=['Fake', 'Real'], digits=4))
        
        cm = confusion_matrix(labels, res['predictions'])
        print("Confusion Matrix:")
        print(f"\n{'':15} {'Predicted Fake':>15} {'Predicted Real':>15}")
        print(f"{'Actual Fake':15} {cm[0,0]:>15} {cm[0,1]:>15}")
        print(f"{'Actual Real':15} {cm[1,0]:>15} {cm[1,1]:>15}")
        print("-" * 50)

    print("\n" + "="*80)
    print("Summary of Accuracies:")
    for res in results_list:
        print(f"  - {res['name']:<35}: {res['accuracy']*100:.4f}%")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Flexible Ensemble for N models (Baseline + Fuzzy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # با 5 مدل:
  python script.py --model_paths m1.pth m2.pth m3.pth m4.pth m5.pth \
    --test_real_dir ./real --test_fake_dir ./fake
        """
    )
    
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                        help='Paths to model checkpoints (at least 2 models required)')
    parser.add_argument('--test_real_dir', type=str, required=True,
                        help='Directory containing real test images')
    parser.add_argument('--test_fake_dir', type=str, required=True,
                        help='Directory containing fake test images')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for inference (default: 256)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--output_prefix', type=str, default='ensemble_comparison',
                        help='Prefix for output files (default: ensemble_comparison)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size - must match fine-tuning size (default: 224)')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=[0.485, 0.456, 0.406],
                        help='Normalization mean values (default: ImageNet)')
    parser.add_argument('--norm_std', type=float, nargs=3, default=[0.229, 0.224, 0.225],
                        help='Normalization std values (default: ImageNet)')
    
    args = parser.parse_args()

    if len(args.model_paths) < 2:
        raise ValueError("At least 2 models are required for ensemble!")
    
    num_models = len(args.model_paths)
    print(f"\n{'='*70}")
    print(f"🎯 Running Ensemble Comparison with {num_models} Models")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.norm_mean, std=args.norm_std)
    ])

    print("\nloading the dataset...")
    test_dataset = WildDeepfakeDataset(
        real_path=args.test_real_dir,
        fake_path=args.test_fake_dir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print(f"\nloading {num_models} models...")
    models = []
    for i, path in enumerate(args.model_paths, 1):
        print(f"\n--- Model {i}/{num_models} ---")
        model = load_pruned_model(path, device)
        models.append(model)

    print(f"\ngetting predictions from all {num_models} models...")
    all_probs = []
    labels = None
    for i, model in enumerate(models, 1):
        print(f"\nModel {i}/{num_models}:")
        probs, lbls = get_predictions(model, test_loader, device)
        all_probs.append(probs)
        if labels is None:
            labels = lbls

    # === شروع بخش اسمبل ===
    all_results = []
    
    # 1. روش اول: میانگین‌گیری ساده (Simple Averaging)
    print("\n" + "="*70)
    print("Running Method 1: Simple Averaging")
    print("="*70)
    simple_preds, simple_acc, simple_name = simple_weighted_ensemble(all_probs, labels, weights=None)
    all_results.append({'name': simple_name, 'predictions': simple_preds, 'accuracy': simple_acc})

    # 2. روش دوم: میانگین‌گیری وزنی (Weighted Averaging)
    print("\n" + "="*70)
    print("Running Method 2: Weighted Averaging")
    print("="*70)
    # محاسبه وزن‌ها بر اساس دقت هر مدل روی دیتاست تست
    print("⚠️  NOTE: Calculating weights on the test set can lead to optimistic results (data leakage).")
    print("    For a robust evaluation, use a separate validation set to determine weights.")
    individual_accs = []
    for idx, probs in enumerate(all_probs):
        preds = np.argmax(probs, axis=1)
        acc = (preds == labels).mean()
        individual_accs.append(acc)
        print(f"  - Model {idx+1} Accuracy (for weight): {acc*100:.2f}%")
    
    weighted_preds, weighted_acc, weighted_name = simple_weighted_ensemble(all_probs, labels, weights=individual_accs)
    all_results.append({'name': weighted_name, 'predictions': weighted_preds, 'accuracy': weighted_acc})

    # 3. روش سوم: اسمبل فازی (Fuzzy Ensemble)
    print("\n" + "="*70)
    print("Running Method 3: Fuzzy Ensemble (Gompertz-based)")
    print("="*70)
    fuzzy_preds, fuzzy_acc, fuzzy_name, fusion_details = fuzzy_ensemble_multi(all_probs, labels)
    all_results.append({'name': fuzzy_name, 'predictions': fuzzy_preds, 'accuracy': fuzzy_acc, 'details': fusion_details})

    # چاپ گزارش نهایی مقایسه‌ای
    print_comparison_report(labels, all_results)

    # ذخیره نتایج
    final_results_dict = {
        'comparison_report': all_results,
        'true_labels': labels,
        'model_probabilities': all_probs,
        'individual_model_accuracies': individual_accs
    }
    
    output_pt = f'{args.output_prefix}_{num_models}models_results.pt'
    torch.save(final_results_dict, output_pt)

    df_dict = {'true_label': labels}
    for res in all_results:
        # Create a safe column name
        col_name = res['name'].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        df_dict[f'{col_name}_prediction'] = res['predictions']
        df_dict[f'{col_name}_is_correct'] = (res['predictions'] == labels).astype(int)

    for i, probs in enumerate(all_probs):
        df_dict[f'model{i+1}_prob_fake'] = probs[:, 0]
        df_dict[f'model{i+1}_prob_real'] = probs[:, 1]

    df_results = pd.DataFrame(df_dict)
    output_csv = f'{args.output_prefix}_{num_models}models_results.csv'
    df_results.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Results saved:")
    print(f"   - PyTorch results: {output_pt}")
    print(f"   - CSV results:      {output_csv}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
