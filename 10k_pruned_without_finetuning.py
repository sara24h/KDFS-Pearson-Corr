import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast
import argparse
from model.pruned_model.Resnet_final import ResNet_50_pruned_hardfakevsreal

WILD_MEAN = [0.4415, 0.3450, 0.3161]
WILD_STD  = [0.2400, 0.2104, 0.2132]

def get_test_transforms():
    """Return test-time transforms for Wild dataset."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=WILD_MEAN, std=WILD_STD)
    ])


class WildDeepfakeDataset(Dataset):
    def __init__(self, real_path, fake_path, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        if os.path.exists(real_path):
            real_files = [f for f in os.listdir(real_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in real_files:
                self.images.append(os.path.join(real_path, fname))
                self.labels.append(1)  # real = 1

        if os.path.exists(fake_path):
            fake_files = [f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in fake_files:
                self.images.append(os.path.join(fake_path, fname))
                self.labels.append(0)  # fake = 0

        print(f"📊 Dataset loaded: {len(self.images)} images "
              f"({sum(1 for l in self.labels if l == 1)} real, {sum(1 for l in self.labels if l == 0)} fake)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            return torch.zeros(3, 256, 256), torch.tensor(label, dtype=torch.float32)


@torch.no_grad()
def test_model(model, loader, device):
    """Test the model and calculate metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    correct = 0
    total = 0
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    print("\n🔍 Starting inference...")
    for inputs, labels in tqdm(loader, desc="Testing"):
        inputs = inputs.to(device)
        labels_np = labels.numpy()
        labels = labels.to(device).unsqueeze(1)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs, _ = model(inputs)
        
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(float)
        
        all_probs.extend(probs.flatten())
        all_preds.extend(preds.flatten())
        all_labels.extend(labels_np)
        
        # Calculate metrics
        for pred, label in zip(preds.flatten(), labels_np):
            if pred == 1 and label == 1:
                true_positives += 1
            elif pred == 0 and label == 0:
                true_negatives += 1
            elif pred == 1 and label == 0:
                false_positives += 1
            elif pred == 0 and label == 1:
                false_negatives += 1
        
        correct += (preds.flatten() == labels_np).sum()
        total += len(labels_np)
    
    # Calculate metrics
    accuracy = 100.0 * correct / total
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1_score * 100,
        'specificity': specificity * 100,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_samples': total,
        'correct_predictions': correct
    }
    
    return results, all_preds, all_labels, all_probs


def main(args):
    print("=" * 70)
    print("    Testing Pruned ResNet50 WITHOUT Fine-tuning")
    print("    Dataset: WildDeepfake")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    print(f"\n📦 Loading pretrained model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    masks_detached = [m.detach().clone() if m is not None else None for m in checkpoint['masks']]
    model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded successfully")
    print(f"   Total parameters: {total_params:,}")
    
    # Prepare test dataset
    print("\n📊 Preparing test dataset...")
    base_path = args.data_path
    test_real = os.path.join(base_path, "test/real")
    test_fake = os.path.join(base_path, "test/fake")
    
    test_transform = get_test_transforms()
    test_dataset = WildDeepfakeDataset(test_real, test_fake, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # Test the model
    results, preds, labels, probs = test_model(model, test_loader, device)
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS (Without Fine-tuning)")
    print("=" * 70)
    print(f"Accuracy:     {results['accuracy']:.2f}%")
    print(f"Precision:    {results['precision']:.2f}%")
    print(f"Recall:       {results['recall']:.2f}%")
    print(f"F1-Score:     {results['f1_score']:.2f}%")
    print(f"Specificity:  {results['specificity']:.2f}%")
    print("\n" + "-" * 70)
    print("Confusion Matrix:")
    print(f"  True Positives:  {results['true_positives']}")
    print(f"  True Negatives:  {results['true_negatives']}")
    print(f"  False Positives: {results['false_positives']}")
    print(f"  False Negatives: {results['false_negatives']}")
    print(f"  Total Samples:   {results['total_samples']}")
    print("=" * 70)
    
    # Save results to JSON
    output_path = args.output_path
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Optionally save predictions
    if args.save_predictions:
        pred_path = output_path.replace('.json', '_predictions.json')
        pred_data = {
            'predictions': [int(p) for p in preds],
            'labels': [int(l) for l in labels],
            'probabilities': [float(p) for p in probs]
        }
        with open(pred_path, 'w') as f:
            json.dump(pred_data, f, indent=4)
        print(f"💾 Predictions saved to: {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pruned ResNet50 without fine-tuning on WildDeepfake")
    parser.add_argument('--model_path', type=str, 
                        default='/kaggle/input/10k-pruned-model/pytorch/default/1/10k_final (1).pt',
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--data_path', type=str, 
                        default='/kaggle/input/wild-deepfake',
                        help='Path to WildDeepfake dataset')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loading workers')
    parser.add_argument('--output_path', type=str, 
                        default='/kaggle/working/test_results_no_finetuning.json',
                        help='Path to save test results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save individual predictions to file')
    
    args = parser.parse_args()
    main(args)
