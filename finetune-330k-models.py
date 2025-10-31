import json
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import argparse
from model.pruned_model.Resnet_final import ResNet_50_pruned_hardfakevsreal

MEAN = [0.4923, 0.4042, 0.3624]
STD = [0.2446, 0.2198, 0.2141]

def get_transforms(is_train=True):

    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
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

        print(f"Dataset loaded: {len(self.images)} images "
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
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 256, 256), torch.tensor(label, dtype=torch.float32)


def create_dataloaders(
    train_real_path,
    train_fake_path,
    val_real_path,
    val_fake_path,
    test_real_path,
    test_fake_path,
    batch_size=256,
    num_workers=4
):
    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)

    train_dataset = WildDeepfakeDataset(train_real_path, train_fake_path, transform=train_transform)
    val_dataset = WildDeepfakeDataset(val_real_path, val_fake_path, transform=val_transform)
    test_dataset = WildDeepfakeDataset(test_real_path, test_fake_path, transform=val_transform)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


def train_epoch(model, loader, criterion, optimizer, device, scaler, writer, epoch, rank=0, accum_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", disable=rank != 0)

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * accum_steps:.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    avg_loss = torch.tensor(running_loss / len(loader)).to(device)
    avg_acc = torch.tensor(100. * correct / total).to(device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    avg_acc = avg_acc.item() / dist.get_world_size()

    if rank == 0 and writer is not None:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/acc", avg_acc, epoch)

    return avg_loss, avg_acc

@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch, rank=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Validation", disable=rank != 0):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = torch.tensor(running_loss / len(loader)).to(device)
    avg_acc = torch.tensor(100. * correct / total).to(device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    avg_acc = avg_acc.item() / dist.get_world_size()

    if rank == 0 and writer is not None:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/acc", avg_acc, epoch)

    return avg_loss, avg_acc

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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
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
    BATCH_SIZE = BATCH_SIZE_PER_GPU * world_size
    NUM_EPOCHS = args.num_epochs
    BASE_LR = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    ACCUM_STEPS = args.accum_steps

    result_dir = f'/kaggle/working/runs_ddp_rank_{global_rank}'
    writer = SummaryWriter(result_dir) if global_rank == 0 else None

    train_real = os.path.join(args.data_path, args.train_real_dir)
    train_fake = os.path.join(args.data_path, args.train_fake_dir)
    val_real = os.path.join(args.data_path, args.val_real_dir)
    val_fake = os.path.join(args.data_path, args.val_fake_dir)
    test_real = os.path.join(args.data_path, args.test_real_dir)
    test_fake = os.path.join(args.data_path, args.test_fake_dir)

    if global_rank == 0:
        print("=" * 70)
        print("    Starting fine-tuning of Pruned ResNet50")
        print(f"   Number of GPUs: {world_size}")
        print(f"   Total Batch Size: {BATCH_SIZE}")
        print(f"   Gradient Accumulation: {ACCUM_STEPS}")
        print(f"   Effective Batch Size: {BATCH_SIZE * ACCUM_STEPS}")
        print(f"   Number of Epochs: {NUM_EPOCHS}")
        print(f"   Learning Rate: {BASE_LR}")
        print(f"   Weight Decay: {WEIGHT_DECAY}")
        print(f"   Dataset path: {args.data_path}")
        print(f"   Input size: 256×256")
        print(f"   Normalization values:")
        print(f"     MEAN: {MEAN}")
        print(f"     STD: {STD}")
        print("=" * 70)

    input_model_path = args.model_path
    checkpoint = torch.load(input_model_path, map_location=DEVICE)
    masks_detached = [m.detach().clone() if m is not None else None for m in checkpoint['masks']]

    model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 and fc for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if global_rank == 0:
        print(f"Model loaded and configured")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Trainable layers: layer4, fc")

    if global_rank == 0:
        print("\nPreparing DataLoaders...")

    train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = create_dataloaders(
        train_real_path=train_real,
        train_fake_path=train_fake,
        val_real_path=val_real,
        val_fake_path=val_fake,
        test_real_path=test_real,
        test_fake_path=test_fake,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=args.num_workers
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW([
        {'params': model.module.layer4.parameters(), 'lr': BASE_LR*0.1, 'weight_decay': WEIGHT_DECAY },
        {'params': model.module.fc.parameters(), 'lr': BASE_LR , 'weight_decay': WEIGHT_DECAY }
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.num_epochs, T_mult=1, eta_min=1e-6
    )
    scaler = GradScaler(enabled=True)

    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, 'best_pruned_finetuned.pt')

    try:
        for epoch in range(NUM_EPOCHS):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            if global_rank == 0:
                print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
                print(f"   LR (layer4): {optimizer.param_groups[0]['lr']:.7f}")
                print(f"   LR (fc):     {optimizer.param_groups[1]['lr']:.7f}")
                print("-" * 70)

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE, scaler, writer,
                epoch, global_rank, ACCUM_STEPS
            )
            if global_rank == 0:
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, writer, epoch, global_rank)
            if global_rank == 0:
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.module.state_dict(), best_model_path)
                    print(f"*** Best model saved with Val Acc: {val_acc:.2f}% ***")

            scheduler.step()

        if global_rank == 0:
            if os.path.exists(best_model_path):
                model.module.load_state_dict(torch.load(best_model_path))
            else:
                print("Warning: Best model not found. Using last epoch weights.")

        test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, writer, NUM_EPOCHS, global_rank)

        if global_rank == 0:
            print("\n" + "=" * 70)
            print("Final test and saving inference-ready model")
            print("=" * 70)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

            model_inference = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])
            model_inference.load_state_dict(model.module.state_dict())
            model_inference = model_inference.cpu().eval()

            total_params_inf = sum(p.numel() for p in model_inference.parameters())

            checkpoint_inference = {
                'model_state_dict': model_inference.state_dict(),
                'total_params': total_params_inf,
                'masks': checkpoint['masks'],
                'model_architecture': 'ResNet_50_pruned_hardfakevsreal (Layer4+FC BCE)',
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'training_config': {
                    'lr': BASE_LR,
                    'weight_decay': WEIGHT_DECAY,
                    'batch_size': BATCH_SIZE,
                    'accum_steps': ACCUM_STEPS,
                    'epochs': NUM_EPOCHS,
                    'loss': 'BCEWithLogitsLoss',
                    'data_path': args.data_path,
                    'input_size': [256, 256],
                    'normalization_mean': MEAN,
                    'normalization_std': STD,
                }
            }

            inference_save_path = os.path.join(args.output_dir, 'final_pruned_finetuned_inference_ready.pt')
            torch.save(checkpoint_inference, inference_save_path)

            print("*** Pruned model successfully saved! ***")
            print(f"Total parameters: {total_params_inf:,}")
            print(f"Best Val Acc: {best_val_acc:.2f}%")
            print(f"Test Acc: {test_acc:.2f}%")

            file_size_mb = os.path.getsize(inference_save_path) / (1024 * 1024)
            print(f"File size: {file_size_mb:.2f} MB")

            if writer:
                writer.close()

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Pruned ResNet50 for Deepfake Detection")

    parser.add_argument('--num_epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay for optimizer')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Root path of the dataset')

    parser.add_argument('--train_real_dir', type=str, default='train/real',
                        help='Relative path to train real images folder')
    parser.add_argument('--train_fake_dir', type=str, default='train/fake',
                        help='Relative path to train fake images folder')
    parser.add_argument('--val_real_dir', type=str, default='validation/real',
                        help='Relative path to validation real images folder')
    parser.add_argument('--val_fake_dir', type=str, default='validation/fake',
                        help='Relative path to validation fake images folder')
    parser.add_argument('--test_real_dir', type=str, default='test/real',
                        help='Relative path to test real images folder')
    parser.add_argument('--test_fake_dir', type=str, default='test/fake',
                        help='Relative path to test fake images folder')

    parser.add_argument('--model_path', type=str,
                        default='/kaggle/input/330k-base-pruned/pytorch/default/1/330k_base_pruned.pt',
                        help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working',
                        help='Directory to save outputs')

    args = parser.parse_args()
    main(args)
