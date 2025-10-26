import json
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast
import argparse
from model.pruned_model.Resnet_final import ResNet_50_pruned_hardfakevsreal

WILD_MEAN = [0.4415, 0.3450, 0.3161]
WILD_STD  = [0.2400, 0.2104, 0.2132]

REALVSFAKE_MEAN = [0.5256, 0.4289, 0.3770]
REALVSFAKE_STD  = [0.2414, 0.2127, 0.2079]


def get_transforms(dataset_name):
    """Return the appropriate transforms for evaluation."""
    if dataset_name == "wild":
        mean, std = WILD_MEAN, WILD_STD
    elif dataset_name == "realvsfake":
        mean, std = REALVSFAKE_MEAN, REALVSFAKE_STD
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Valid options: 'wild', 'realvsfake'")

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
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


def create_dataloader(real_path, fake_path, dataset_name, batch_size=128, num_workers=4):
    """Create a single dataloader for testing."""
    transform = get_transforms(dataset_name)
    dataset = WildDeepfakeDataset(real_path, fake_path, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=2, 
        pin_memory=True, 
        drop_last=False
    )
    return loader, sampler


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, rank=0):
    """Evaluate model and return detailed metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    
    for inputs, labels in tqdm(loader, desc="Testing", disable=rank != 0):
        inputs, labels = inputs.to(device), labels.to(device)
        labels_unsqueezed = labels.unsqueeze(1)
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels_unsqueezed)
        
        running_loss += loss.item()
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs > 0.5).float()
        
        # Convert to float32 before numpy conversion (bfloat16 not supported by numpy)
        all_preds.extend(preds.float().cpu().numpy())
        all_labels.extend(labels.float().cpu().numpy())
        all_probs.extend(probs.float().cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    avg_loss = running_loss / len(loader)
    
    # Confusion matrix
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'specificity': specificity * 100,
        'confusion_matrix': {
            'TP': int(tp), 'TN': int(tn),
            'FP': int(fp), 'FN': int(fn)
        }
    }
    
    # Aggregate across GPUs
    for key in ['loss', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
        value_tensor = torch.tensor(metrics[key]).to(device)
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        metrics[key] = value_tensor.item() / dist.get_world_size()
    
    return metrics


def setup_ddp(seed):
    os.environ['TORCH_NCCL_TIMEOUT_MS'] = '1800000'
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    seed = seed + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    SEED = 42
    local_rank = setup_ddp(SEED)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    DEVICE = torch.device(f"cuda:{local_rank}")
    BATCH_SIZE_PER_GPU = args.batch_size

    if global_rank == 0:
        print("=" * 80)
        print("        🧪 MODEL EVALUATION - CROSS-DATASET TESTING")
        print("=" * 80)
        print(f"   🎯 Test Dataset: {args.test_dataset}")
        print(f"   🔧 Model Path: {args.model_path}")
        print(f"   💻 Number of GPUs: {world_size}")
        print(f"   📦 Batch Size per GPU: {BATCH_SIZE_PER_GPU}")
        print("=" * 80)

    # Load model
    if global_rank == 0:
        print("\n🔄 Loading model...")
    
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    
    if 'masks' in checkpoint:
        masks = checkpoint['masks']
    else:
        masks = None
    
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if global_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully")
        print(f"   Total parameters: {total_params:,}")
        if 'best_val_acc' in checkpoint:
            print(f"   Model's best validation accuracy: {checkpoint['best_val_acc']:.2f}%")

    # Setup test dataset paths
    if args.test_dataset == "wild":
        base_path = "/kaggle/input/wild-deepfake"
        test_real = os.path.join(base_path, "test/real")
        test_fake = os.path.join(base_path, "test/fake")
    elif args.test_dataset == "realvsfake":
        base_path = "/kaggle/input/realvsfake/whole"
        test_real = os.path.join(base_path, "test/test_real")
        test_fake = os.path.join(base_path, "test/test_fake")
    else:
        raise ValueError("Test dataset must be either 'wild' or 'realvsfake'.")

    if global_rank == 0:
        print(f"\n📊 Loading test dataset: {args.test_dataset}")
    
    test_loader, test_sampler = create_dataloader(
        real_path=test_real,
        fake_path=test_fake,
        dataset_name=args.test_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=4
    )

    criterion = nn.BCEWithLogitsLoss()

    try:
        if global_rank == 0:
            print("\n🚀 Starting evaluation...")
            print("-" * 80)
        
        test_sampler.set_epoch(0)
        metrics = evaluate_model(model, test_loader, criterion, DEVICE, global_rank)

        if global_rank == 0:
            print("\n" + "=" * 80)
            print("📊 EVALUATION RESULTS")
            print("=" * 80)
            print(f"Test Dataset: {args.test_dataset}")
            print(f"Model: {args.model_path}")
            print("-" * 80)
            print(f"🎯 Accuracy:    {metrics['accuracy']:.2f}%")
            print(f"📉 Loss:        {metrics['loss']:.4f}")
            print(f"🎪 Precision:   {metrics['precision']:.2f}%")
            print(f"🔍 Recall:      {metrics['recall']:.2f}%")
            print(f"⚖️  F1-Score:    {metrics['f1_score']:.2f}%")
            print(f"✨ Specificity: {metrics['specificity']:.2f}%")
            print("-" * 80)
            print("Confusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"   True Positives:  {cm['TP']:6d}  |  False Positives: {cm['FP']:6d}")
            print(f"   False Negatives: {cm['FN']:6d}  |  True Negatives:  {cm['TN']:6d}")
            print("=" * 80)

            # Save results
            results = {
                'test_dataset': args.test_dataset,
                'model_path': args.model_path,
                'test_metrics': metrics,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            result_filename = f'/kaggle/working/test_results_{args.test_dataset}.json'
            with open(result_filename, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"\n💾 Results saved to: {result_filename}")
            
            # Performance interpretation
            print("\n📈 PERFORMANCE ANALYSIS:")
            
            if metrics['accuracy'] >= 90:
                print("   ✅ EXCELLENT performance!")
            elif metrics['accuracy'] >= 80:
                print("   ✔️  GOOD performance")
            elif metrics['accuracy'] >= 70:
                print("   ⚠️  MODERATE performance")
            else:
                print("   ❌ POOR performance")
            
            print("=" * 80)

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Model on Dataset")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--test_dataset', type=str, required=True, 
                        choices=['wild', 'realvsfake'],
                        help="Dataset to test the model on")
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size per GPU')
    args = parser.parse_args()
    
    main(args)
