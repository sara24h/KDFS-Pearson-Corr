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
    return np.array(predictions), accuracy, fusion_details


def print_detailed_results(labels, predictions, model_probs_list):
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n" + "="*70)
    print(classification_report(labels, predictions, target_names=['Fake', 'Real'], digits=4))
    
    print("\n" + "="*70)
    print("confusion matrix:")
    print("="*70)
    cm = confusion_matrix(labels, predictions)
    print(f"\n{'':15} {'Predicted Fake':>15} {'Predicted Real':>15}")
    print(f"{'Actual Fake':15} {cm[0,0]:>15} {cm[0,1]:>15}")
    print(f"{'Actual Real':15} {cm[1,0]:>15} {cm[1,1]:>15}")
    
    print("\n" + "="*70)
    print(f"✅ Fake correctly classified: {cm[0,0]} / {cm[0,0] + cm[0,1]}")
    print(f"❌ Fake misclassified as Real: {cm[0,1]} / {cm[0,0] + cm[0,1]}")
    print(f"✅ Real correctly classified: {cm[1,1]} / {cm[1,0] + cm[1,1]}")
    print(f"❌ Real misclassified as Fake: {cm[1,0]} / {cm[1,0] + cm[1,1]}")
    
    print("\n" + "="*70)
    print("Individual Model Performance:")
    print("="*70)
    individual_accs = []
    for idx, probs in enumerate(model_probs_list):
        preds = np.argmax(probs, axis=1)
        acc = (preds == labels).mean()
        individual_accs.append(acc)
        print(f"Model {idx+1} Accuracy: {acc*100:.2f}%")
    
    ensemble_acc = (predictions == labels).mean()
    best_single = max(individual_accs)
    improvement = (ensemble_acc - best_single) * 100
    print(f"\n{'='*70}")
    print(f"Fuzzy Ensemble Accuracy: {ensemble_acc*100:.2f}%")
    print(f"Best Single Model Accuracy: {best_single*100:.2f}%")
    print(f"Improvement over best single model: {improvement:+.2f}%")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Flexible Fuzzy Ensemble for N models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # با 2 مدل (سایز پیش‌فرض 224×224):
  python script.py --model_paths model1.pth model2.pth --test_real_dir ./real --test_fake_dir ./fake
  
  # با 5 مدل و سایز سفارشی:
  python script.py --model_paths m1.pth m2.pth m3.pth m4.pth m5.pth \
    --test_real_dir ./real --test_fake_dir ./fake --input_size 224
  
  # با normalization سفارشی (دیتاست خودتان):
  python script.py --model_paths m1.pth m2.pth \
    --test_real_dir ./real --test_fake_dir ./fake \
    --input_size 224 \
    --norm_mean 0.3594 0.3140 0.3242 \
    --norm_std 0.2499 0.2249 0.2268
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
    parser.add_argument('--output_prefix', type=str, default='fuzzy_ensemble',
                        help='Prefix for output files (default: fuzzy_ensemble)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size - must match fine-tuning size (default: 224)')
    parser.add_argument('--norm_mean', type=float, nargs=3, default=[0.485, 0.456, 0.406],
                        help='Normalization mean values (default: ImageNet)')
    parser.add_argument('--norm_std', type=float, nargs=3, default=[0.229, 0.224, 0.225],
                        help='Normalization std values (default: ImageNet)')
    
    args = parser.parse_args()

    # Validate number of models
    if len(args.model_paths) < 2:
        raise ValueError("At least 2 models are required for ensemble!")
    
    num_models = len(args.model_paths)
    print(f"\n{'='*70}")
    print(f"🎯 Fuzzy Ensemble with {num_models} Models (Paper Method)")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    # Transform باید دقیقاً مثل زمان fine-tune باشد
    print(f"\n⚙️  Transform Settings:")
    print(f"   - Input Size: {args.input_size}×{args.input_size}")
    print(f"   - Normalize Mean: {args.norm_mean}")
    print(f"   - Normalize Std: {args.norm_std}")
    
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

    print("\n" + "="*70)
    print(f"ensembeling {num_models} models with fuzzy logic (Gompertz-based)...")
    print("="*70)
    final_predictions, accuracy, fusion_details = fuzzy_ensemble_multi(all_probs, labels)

    print(f"\naccuracy of fuzzy ensemble: {accuracy * 100:.2f}%")
    print_detailed_results(labels, final_predictions, all_probs)

    results = {
        'final_predictions': final_predictions,
        'true_labels': labels,
        'accuracy': accuracy,
        'num_models': num_models,
        'model_paths': args.model_paths,
        'model_probabilities': all_probs,
        'fusion_details': fusion_details[:100],
        'dataset_info': {
            'total_samples': len(labels),
            'real_samples': int((labels == 1).sum()),  # Adjusted for label 1=real
            'fake_samples': int((labels == 0).sum())
        }
    }
    
    output_pt = f'{args.output_prefix}_{num_models}models_results.pt'
    torch.save(results, output_pt)

    df_dict = {
        'true_label': labels,
        'fuzzy_prediction': final_predictions,
        'is_correct': (final_predictions == labels).astype(int)
    }
    
    for i, probs in enumerate(all_probs):
        df_dict[f'model{i+1}_prob_fake'] = probs[:, 0]  # class 0: fake
        df_dict[f'model{i+1}_prob_real'] = probs[:, 1]  # class 1: real

    df_results = pd.DataFrame(df_dict)
    output_csv = f'{args.output_prefix}_{num_models}models_results.csv'
    df_results.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Results saved:")
    print(f"   - {output_pt}")
    print(f"   - {output_csv}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
