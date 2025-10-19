import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import argparse
from model.pruned_model.Resnet_final import ResNet_50_pruned_hardfakevsreal

class DeepfakeDataset(Dataset):
    def __init__(self, real_dirs, fake_dirs, transform=None):
        self.image_paths = []
        self.labels = []
        
        # Load real images
        for dir_path in real_dirs:
            paths = glob.glob(os.path.join(dir_path, "*.jpg")) + \
                   glob.glob(os.path.join(dir_path, "*.png"))
            self.image_paths.extend(paths)
            self.labels.extend([1] * len(paths))
        
        # Load fake images
        for dir_path in fake_dirs:
            paths = glob.glob(os.path.join(dir_path, "*.jpg")) + \
                   glob.glob(os.path.join(dir_path, "*.png"))
            self.image_paths.extend(paths)
            self.labels.extend([0] * len(paths))
        
        self.transform = transform
        print(f"📊 Dataset loaded: {len(self.image_paths)} images "
              f"({sum(self.labels)} real, {len(self.labels) - sum(self.labels)} fake)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float32)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_one_epoch(model, dataloader, criterion, optimizer, device, rank, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device, rank):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def test_model(model, dataloader, criterion, device, rank):
    """Test the model and return detailed metrics"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    if rank == 0:
        print("\n🧪 Testing model on test dataset...")
        pbar = tqdm(dataloader, desc="Testing")
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Calculate confusion matrix elements
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            true_negatives += ((predictions == 0) & (labels == 0)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def main_worker(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Starting DDP Training on {world_size} GPUs")
        print(f"{'='*70}\n")

    train_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4414, 0.3448, 0.3159], std=[0.1854, 0.1623, 0.1562])
    ])
    
    val_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4414, 0.3448, 0.3159], std=[0.1854, 0.1623, 0.1562])
    ])

    # Train dataset
    if rank == 0:
        print("Loading training dataset...")
    train_dataset = DeepfakeDataset(
        real_dirs=["/kaggle/input/wild-deepfake/train/real"],
        fake_dirs=["/kaggle/input/wild-deepfake/train/fake"],
        transform=train_transform
    )
    
    # Validation dataset
    if rank == 0:
        print("Loading validation dataset...")
    val_dataset = DeepfakeDataset(
        real_dirs=["/kaggle/input/wild-deepfake/valid/real"],
        fake_dirs=["/kaggle/input/wild-deepfake/valid/fake"],
        transform=val_transform
    )
    
    # Test dataset
    if rank == 0:
        print("Loading test dataset...")
    test_dataset = DeepfakeDataset(
        real_dirs=["/kaggle/input/wild-deepfake/test/real"],
        fake_dirs=["/kaggle/input/wild-deepfake/test/fake"],
        transform=val_transform
    )
    
    # Samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"\nLoading pretrained model from {args.model_path}...")
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    masks = checkpoint['masks']
    
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   ✅ Model loaded successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}\n")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*70}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, rank, epoch
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, rank)
        
        scheduler.step()
        
        if rank == 0:
            print(f"\n📊 Epoch {epoch} Results:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(args.output_dir, 'best_model_finetuned.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'masks': masks,
                    'val_acc': val_acc,
                    'train_acc': train_acc
                }, save_path)
                print(f"   💾 New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Test the best model
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Loading best model for testing...")
        print(f"{'='*70}")
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, 'best_model_finetuned.pt')
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    
    # Test on test dataset
    test_results = test_model(model, test_loader, criterion, device, rank)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"🎯 FINAL TEST RESULTS")
        print(f"{'='*70}")
        print(f"   Test Loss: {test_results['loss']:.4f}")
        print(f"   Test Accuracy: {test_results['accuracy']:.2f}%")
        print(f"   Precision: {test_results['precision']:.4f}")
        print(f"   Recall: {test_results['recall']:.4f}")
        print(f"   F1 Score: {test_results['f1_score']:.4f}")
        print(f"\n📊 Confusion Matrix:")
        print(f"   True Positives (Real detected as Real): {test_results['true_positives']}")
        print(f"   True Negatives (Fake detected as Fake): {test_results['true_negatives']}")
        print(f"   False Positives (Fake detected as Real): {test_results['false_positives']}")
        print(f"   False Negatives (Real detected as Fake): {test_results['false_negatives']}")
        print(f"{'='*70}")
        
        print(f"\n✅ Training completed!")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        print(f"   Test accuracy: {test_results['accuracy']:.2f}%")
        print(f"{'='*70}\n")
        
        # Save test results
        results_path = os.path.join(args.output_dir, 'test_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Test Results\n")
            f.write(f"{'='*50}\n")
            f.write(f"Test Loss: {test_results['loss']:.4f}\n")
            f.write(f"Test Accuracy: {test_results['accuracy']:.2f}%\n")
            f.write(f"Precision: {test_results['precision']:.4f}\n")
            f.write(f"Recall: {test_results['recall']:.4f}\n")
            f.write(f"F1 Score: {test_results['f1_score']:.4f}\n")
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"True Positives: {test_results['true_positives']}\n")
            f.write(f"True Negatives: {test_results['true_negatives']}\n")
            f.write(f"False Positives: {test_results['false_positives']}\n")
            f.write(f"False Negatives: {test_results['false_negatives']}\n")
        
        print(f"📝 Test results saved to: {results_path}")
    
    cleanup()
  
def main():
    parser = argparse.ArgumentParser(description='DDP Training for Pruned ResNet50')
    parser.add_argument('--model_path', type=str, default='/kaggle/working/10k_final.pt',
                        help='Path to the pretrained pruned model')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--world_size', type=int, default=2,
                        help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    torch.multiprocessing.spawn(
        main_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == '__main__':
    main()
