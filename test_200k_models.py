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

MEAN = [0.4868, 0.3972, 0.3624]
STD = [0.2296, 0.2066, 0.2009]

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


def create_test_dataloader(
    test_real_path,
    test_fake_path,
    batch_size=256,
    num_workers=4
):
    test_transform = get_transforms(is_train=False)

    test_dataset = WildDeepfakeDataset(test_real_path, test_fake_path, transform=test_transform)

    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return test_loader, test_sampler


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
        writer.add_scalar("test/loss", avg_loss, epoch)
        writer.add_scalar("test/acc", avg_acc, epoch)

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

    result_dir = f'/kaggle/working/runs_ddp_rank_{global_rank}'
    writer = SummaryWriter(result_dir) if global_rank == 0 else None

    # مسیرهای داده
    test_real = os.path.join(args.data_path, args.test_real_dir)
    test_fake = os.path.join(args.data_path, args.test_fake_dir)

    if global_rank == 0:
        print("=" * 70)
        print("    Starting Testing Only for Pruned ResNet50")
        print(f"   Number of GPUs: {world_size}")
        print(f"   Dataset path: {args.data_path}")
        print(f"   Input size: 256×256")
        print(f"   Normalization values:")
        print(f"     MEAN: {MEAN}")
        print(f"     STD: {STD}")
        print("=" * 70)

    # بارگذاری مدل
    input_model_path = args.model_path
    checkpoint = torch.load(input_model_path, map_location=DEVICE)
    masks_detached = [m.detach().clone() if m is not None else None for m in checkpoint['masks']]

    model = ResNet_50_pruned_hardfakevsreal(masks=masks_detached)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)

    # استفاده از DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if global_rank == 0:
        print(f"Model loaded for testing")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print("   - Note: All layers are frozen for testing only.")

    # ایجاد دیتالودر تست
    if global_rank == 0:
        print("\nPreparing Test DataLoader...")

    test_loader, test_sampler = create_test_dataloader(
        test_real_path=test_real,
        test_fake_path=test_fake,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    criterion = nn.BCEWithLogitsLoss()

    # تست نهایی
    if global_rank == 0:
        print("\nStarting Test Evaluation...")
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE, writer, epoch=0, rank=global_rank)

    if global_rank == 0:
        print("\n" + "=" * 70)
        print("Test Evaluation Complete")
        print("=" * 70)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        # اگر بخواهید مدل نهایی را دوباره ذخیره کنید (مثلاً فقط برای اطمینان از وضعیت تست)
        # توجه: این مدل دقیقاً همان مدل ورودی است، فقط برای نمایش ساختار متا اضافه می‌کنیم
        model_inference = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])
        model_inference.load_state_dict(model.module.state_dict()) # اگر مدل تغییر کرده باشد (که نیست)، ولی برای اطمینان
        model_inference = model_inference.cpu().eval()

        total_params_inf = sum(p.numel() for p in model_inference.parameters())

        checkpoint_inference = {
            'model_state_dict': model_inference.state_dict(),
            'total_params': total_params_inf,
            'masks': checkpoint['masks'],
            'model_architecture': 'ResNet_50_pruned_hardfakevsreal (Test Only)',
            'test_acc': test_acc,
            'training_config': {
                'data_path': args.data_path,
                'input_size': [256, 256],
                'normalization_mean': MEAN,
                'normalization_std': STD,
            }
        }

        inference_save_path = os.path.join(args.output_dir, 'test_only_inference_ready.pt')
        torch.save(checkpoint_inference, inference_save_path)

        print("*** Test-only model saved (if needed) ***")
        print(f"Total parameters: {total_params_inf:,}")
        print(f"Test Acc: {test_acc:.2f}%")

        file_size_mb = os.path.getsize(inference_save_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

        if writer:
            writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Only Pruned ResNet50 for Deepfake Detection")

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Root path of the dataset')

    parser.add_argument('--test_real_dir', type=str, default='test/real',
                        help='Relative path to test real images folder')
    parser.add_argument('--test_fake_dir', type=str, default='test/fake',
                        help='Relative path to test fake images folder')

    parser.add_argument('--model_path', type=str,
                        default='/kaggle/input/200k-base-pruned/pytorch/default/1/200k_base_pruned.pt',
                        help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working',
                        help='Directory to save outputs')

    args = parser.parse_args()
    main(args)
