import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
import numpy as np
import os
import random
from pathlib import Path
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_seed(42)

def worker_init_fn(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class UADFVDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, image_size=256,
                 transform=None, sampling_strategy='uniform',
                 split='train', split_ratio=(0.7, 0.15, 0.15), seed=42, video_list=None):
       
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.sampling_strategy = sampling_strategy
        self.split = split
        self.seed = seed
        self.video_list = video_list if video_list is not None else self._load_and_split(split_ratio)
        
        # اگر transform داده نشده، بر اساس split تصمیم‌گیری می‌شود
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:  # val / test / full
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        print(f"[{split.upper() if split else 'CUSTOM'}] {len(self.video_list)} videos loaded.")

    def _load_and_split(self, split_ratio):
        video_list = []
        # Fake → label 0
        for p in sorted((self.root_dir / 'fake').glob('*.mp4')):
            if not p.name.startswith('.'):
                video_list.append((str(p), 0))
        # Real → label 1
        for p in sorted((self.root_dir / 'real').glob('*.mp4')):
            if not p.name.startswith('.'):
                video_list.append((str(p), 1))
        
        # Shuffle ثابت
        rng = random.Random(self.seed)
        rng.shuffle(video_list)
        
        total = len(video_list)
        if self.split == 'full':
            return video_list
        
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])
        
        if self.split == 'train':
            return video_list[:train_end]
        elif self.split == 'val':
            return video_list[train_end:val_end]
        elif self.split == 'test':
            return video_list[val_end:]
        else:
            raise ValueError("split must be train/val/test/full")

    def sample_frames(self, total_frames: int):
        if total_frames <= self.num_frames:
            idxs = np.random.choice(total_frames, self.num_frames, replace=True)
            return sorted(idxs.tolist())
        
        if self.sampling_strategy == 'uniform':
            return np.linspace(0, total_frames-1, self.num_frames, dtype=int).tolist()
        elif self.sampling_strategy == 'random':
            idxs = np.random.choice(total_frames, self.num_frames, replace=False)
            return sorted(idxs.tolist())
        elif self.sampling_strategy == 'first':
            return list(range(self.num_frames))
        else:
            raise ValueError("sampling_strategy: uniform / random / first")

    def load_video(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {path}")
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self.sample_frames(total)
        
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    frame = self.transform(frame)
                except Exception as e:
                    print(f"Transform error on frame from {path}: {e}")
                    frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(frame)
            else:
                # fallback
                fallback = frames[-1].clone() if frames else torch.zeros(3, self.image_size, self.image_size)
                frames.append(fallback)
        
        cap.release()
        return torch.stack(frames)  # [T, C, H, W]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        path, label = self.video_list[idx]
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = self.seed + worker_info.id * 100000 + idx
        else:
            seed = self.seed + idx
        
        r_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            frames = self.load_video(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        finally:
            random.setstate(r_state)
            np.random.set_state(np_state)
        
        return frames, torch.tensor(label, dtype=torch.float32)


def create_kfold_dataloaders(
    root_dir,
    num_frames=16,
    image_size=256,
    train_batch_size=8,
    eval_batch_size=16,
    val_batch_size=None,  # برای سازگاری با کد قدیمی
    num_workers=4,
    pin_memory=True,
    ddp=False,
    sampling_strategy='uniform',
    split_ratio=(0.7, 0.15, 0.15),
    seed=42,
    n_splits=5
):
    """
    ایجاد K-Fold Cross Validation dataloaders + test set
    
    Returns:
        tuple: (fold_loaders, test_loader)
            - fold_loaders: لیست از (fold_idx, train_loader, val_loader)
            - test_loader: test dataloader مشترک
    """
    # Handle val_batch_size parameter
    if val_batch_size is not None and val_batch_size != eval_batch_size:
        print(f"Warning: val_batch_size ({val_batch_size}) provided but using eval_batch_size ({eval_batch_size})")
    
    # بارگذاری کامل dataset
    temp_ds = UADFVDataset(
        root_dir, num_frames, image_size,
        transform=None, sampling_strategy=sampling_strategy,
        split='full', split_ratio=split_ratio, seed=seed
    )
    
    video_list = temp_ds.video_list
    total = len(video_list)
    
    # تقسیم به train+val و test
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])
    
    trainval_list = video_list[:val_end]
    test_list = video_list[val_end:]
    
    # استخراج labels برای StratifiedKFold
    labels_trainval = np.array([label for _, label in trainval_list])
    
    # ایجاد K-Fold splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_loaders = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(trainval_list)), labels_trainval)):
        print(f"\n{'='*60}")
        print(f"Preparing Fold {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}")
        
        # ساخت video lists برای این fold
        train_videos = [trainval_list[i] for i in train_idx]
        val_videos = [trainval_list[i] for i in val_idx]
        
        print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")
        
        # ایجاد datasets
        train_ds = UADFVDataset(
            root_dir, num_frames, image_size,
            transform=None, sampling_strategy=sampling_strategy,
            split='train', split_ratio=split_ratio, seed=seed,
            video_list=train_videos
        )
        
        val_ds = UADFVDataset(
            root_dir, num_frames, image_size,
            transform=None, sampling_strategy=sampling_strategy,
            split='val', split_ratio=split_ratio, seed=seed,
            video_list=val_videos
        )
        
        # ایجاد samplers
        if ddp:
            train_sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
            val_sampler = DistributedSampler(val_ds, shuffle=False, seed=seed)
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        
        g = torch.Generator().manual_seed(seed + fold_idx)
        
        # ایجاد dataloaders
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            generator=g
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
        
        fold_loaders.append((fold_idx, train_loader, val_loader))
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ایجاد test loader
    print(f"\n{'='*60}")
    print("Preparing Test Set")
    print(f"{'='*60}")
    
    test_ds = UADFVDataset(
        root_dir, num_frames, image_size,
        transform=None, sampling_strategy=sampling_strategy,
        split='test', split_ratio=split_ratio, seed=seed,
        video_list=test_list
    )
    
    if ddp:
        test_sampler = DistributedSampler(test_ds, shuffle=False, seed=seed)
    else:
        test_sampler = None
    
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn
    )
    
    print(f"Test videos: {len(test_list)}, Test batches: {len(test_loader)}")
    print(f"{'='*60}\n")
    
    return fold_loaders, test_loader


if __name__ == "__main__":
    root_dir = "/kaggle/input/uadfv-dataset/UADFV"
    
    fold_loaders, test_loader = create_kfold_dataloaders(
        root_dir=root_dir,
        num_frames=16,
        image_size=256,
        train_batch_size=4,
        eval_batch_size=8,
        num_workers=4,
        pin_memory=True,
        ddp=False,
        sampling_strategy='uniform',
        split_ratio=(0.7, 0.15, 0.15),
        seed=42,
        n_splits=5
    )
    
    print(f"\n{'='*60}")
    print("Testing Data Loaders")
    print(f"{'='*60}\n")
    
    # تست هر fold
    for fold_idx, train_loader, val_loader in fold_loaders:
        print(f"\nTesting Fold {fold_idx + 1}:")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # تست یک batch از train
        for videos, labels in train_loader:
            print(f"  Train batch shape: {videos.shape}")  # [B, T, C, H, W]
            print(f"  Train labels: {labels.tolist()}")
            break
        
        # تست یک batch از val
        for videos, labels in val_loader:
            print(f"  Val batch shape: {videos.shape}")
            print(f"  Val labels: {labels.tolist()}")
            break
    
    # تست test loader
    print(f"\nTest batches: {len(test_loader)}")
    for videos, labels in test_loader:
        print(f"Test batch shape: {videos.shape}")
        print(f"Test labels: {labels.tolist()}")
        break
