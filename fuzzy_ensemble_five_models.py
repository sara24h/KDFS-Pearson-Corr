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
                self.labels.append(0)
        else:
            raise FileNotFoundError(f"real folder not found: {real_path}")
        
        if os.path.exists(fake_path):
            fake_files = [f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in fake_files:
                self.images.append(os.path.join(fake_path, fname))
                self.labels.append(1)
        else:
            raise FileNotFoundError(f"fake folder not found: {fake_path}")
        
        print(f"number of Real images: {len([l for l in self.labels if l==0])}")
        print(f"number of Fake images: {len([l for l in self.labels if l==1])}")
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
            return torch.zeros(3, 224, 224), label, img_path


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


def fuzzy_ensemble_multi(model_probs_list, labels, class_no=2):
    """
    Perform fuzzy ensemble over N models (N >= 2).
    model_probs_list: list of numpy arrays, each of shape (N_samples, 2)
    """
    num_models = len(model_probs_list)
    num_samples = len(labels)
    correct = 0
    predictions = []
    fusion_details = []

    for i in range(num_samples):
        rank_sum = np.zeros((class_no, 1))
        score_sum = np.zeros(class_no)

        for probs in model_probs_list:
            rank1 = generateRank1(probs[i], class_no)
            rank2 = generateRank2(probs[i], class_no)
            rank_combined = rank1 * rank2
            rank_sum += rank_combined
            score_sum += probs[i]

        score_avg = score_sum / num_models
        fused_score = (rank_sum.T) * (1 - score_avg)  # shape: (1, class_no)
        cls = np.argmin(rank_sum)  # or np.argmax(fused_score) — but original uses rankSum min
        predictions.append(cls)

        fusion_details.append({
            'sample_idx': i,
            'rank_sum': rank_sum.flatten(),
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
    print(classification_report(labels, predictions, target_names=['Real', 'Fake'], digits=4))
    
    print("\n" + "="*70)
    print("confusion matrix:")
    print("="*70)
    cm = confusion_matrix(labels, predictions)
    print(f"\n{'':15} {'Predicted Real':>15} {'Predicted Fake':>15}")
    print(f"{'Actual Real':15} {cm[0,0]:>15} {cm[0,1]:>15}")
    print(f"{'Actual Fake':15} {cm[1,0]:>15} {cm[1,1]:>15}")
    
    print("\n" + "="*70)
    print(f"✅ Real correctly classified: {cm[0,0]} / {cm[0,0] + cm[0,1]}")
    print(f"❌ Real misclassified as Fake: {cm[0,1]} / {cm[0,0] + cm[0,1]}")
    print(f"✅ Fake correctly classified: {cm[1,1]} / {cm[1,0] + cm[1,1]}")
    print(f"❌ Fake misclassified as Real: {cm[1,0]} / {cm[1,0] + cm[1,1]}")
    
    individual_accs = []
    for idx, probs in enumerate(model_probs_list):
        preds = np.argmax(probs, axis=1)
        acc = (preds == labels).mean()
        individual_accs.append(acc)
        print(f"Model {idx+1} Accuracy: {acc*100:.2f}%")
    
    ensemble_acc = (predictions == labels).mean()
    best_single = max(individual_accs)
    improvement = (ensemble_acc - best_single) * 100
    print(f"Fuzzy Ensemble Accuracy: {ensemble_acc*100:.2f}%")
    print(f"Improvement over best single model: {improvement:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Fuzzy Ensemble for 5 models")
    parser.add_argument('--model1_path', type=str, required=True)
    parser.add_argument('--model2_path', type=str, required=True)
    parser.add_argument('--model3_path', type=str, required=True)
    parser.add_argument('--model4_path', type=str, required=True)
    parser.add_argument('--model5_path', type=str, required=True)
    parser.add_argument('--test_real_dir', type=str, required=True)
    parser.add_argument('--test_fake_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3594, 0.3140, 0.3242], std=[0.2499, 0.2249, 0.2268])
    ])

    print("\nloading the dataset...")
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
        drop_last=False
    )

    model_paths = [
        args.model1_path,
        args.model2_path,
        args.model3_path,
        args.model4_path,
        args.model5_path
    ]

    print("\nloading models...")
    models = [load_pruned_model(path, device) for path in model_paths]

    print("getting predictions from all models...")
    all_probs = []
    labels = None
    for i, model in enumerate(models, 1):
        probs, lbls = get_predictions(model, test_loader, device)
        all_probs.append(probs)
        if labels is None:
            labels = lbls

    print("\n" + "="*70)
    print("ensembeling 5 models with fuzzy logic...")
    print("="*70)
    final_predictions, accuracy, fusion_details = fuzzy_ensemble_multi(all_probs, labels)

    print(f"\naccuracy of fuzzy ensemble: {accuracy * 100:.2f}%")
    print_detailed_results(labels, final_predictions, all_probs)

    # Save results
    results = {
        'final_predictions': final_predictions,
        'true_labels': labels,
        'accuracy': accuracy,
        'model_probabilities': all_probs,  # list of 5 arrays
        'fusion_details': fusion_details[:100],
        'dataset_info': {
            'total_samples': len(labels),
            'real_samples': int((labels == 0).sum()),
            'fake_samples': int((labels == 1).sum())
        }
    }
    torch.save(results, 'fuzzy_ensemble_5models_results.pt')

    df_dict = {
        'true_label': labels,
        'fuzzy_prediction': final_predictions,
        'is_correct': (final_predictions == labels).astype(int)
    }
    for i, probs in enumerate(all_probs):
        df_dict[f'model{i+1}_prob_real'] = probs[:, 0]
        df_dict[f'model{i+1}_prob_fake'] = probs[:, 1]

    df_results = pd.DataFrame(df_dict)
    df_results.to_csv('fuzzy_ensemble_5models_results.csv', index=False)
    print("\nresults saved: fuzzy_ensemble_5models_results.pt and fuzzy_ensemble_5models_results.csv")


if __name__ == "__main__":
    main()
