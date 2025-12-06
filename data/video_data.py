# data/video_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import cv2
import numpy as np
import random
import os
from pathlib import Path
from torchvision import transforms
from sklearn.model_selection import KFold, StratifiedKFold

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class UADFVDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, image_size=256,
                 transform=None, sampling_strategy='uniform',
                 is_training=True, seed=42):
       
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.sampling_strategy = sampling_strategy
        self.is_training = is_training
        self.seed = seed
        
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else: # validation
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
            
        self.video_list = self._load_videos()
        # چاپ فقط در رنک 0 برای جلوگیری از شلوغی در DDP
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"Total {len(self.video_list)} videos loaded.")

    def _load_videos(self):
        video_list = []
        # Fake → label 0
        fake_dir = self.root_dir / 'fake'
        if fake_dir.exists():
            for p in sorted(fake_dir.glob('*.mp4')):
                if not p.name.startswith('.'):
                    video_list.append((str(p), 0))
        # Real → label 1
        real_dir = self.root_dir / 'real'
        if real_dir.exists():
            for p in sorted(real_dir.glob('*.mp4')):
                if not p.name.startswith('.'):
                    video_list.append((str(p), 1))
        rng = random.Random(self.seed)
        rng.shuffle(video_list)
        return video_list

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
        return torch.stack(frames) # [T, C, H, W]

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
            np.random.setstate(np_state)
        return frames, torch.tensor(label, dtype=torch.float32)

def create_kfold_dataloaders(
    root_dir,
    n_splits=5,
    num_frames=16,
    image_size=256,
    train_batch_size=8,
    val_batch_size=16,
    num_workers=4,
    pin_memory=True,
    ddp=False,
    sampling_strategy='uniform',
    stratified=True,
    seed=42
):
    full_dataset = UADFVDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        image_size=image_size,
        sampling_strategy=sampling_strategy,
        is_training=True,
        seed=seed
    )
    labels = [label for _, label in full_dataset.video_list]
    if stratified:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = kfold.split(np.zeros(len(labels)), labels)
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = kfold.split(np.zeros(len(labels)))
   
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        
        train_dataset = Subset(full_dataset, train_indices)
        
        # Create a separate validation dataset with is_training=False
        val_dataset_full = UADFVDataset(
            root_dir=root_dir,
            num_frames=num_frames,
            image_size=image_size,
            sampling_strategy=sampling_strategy,
            is_training=False, # validation mode
            seed=seed
        )
        val_dataset = Subset(val_dataset_full, val_indices)
       
        if ddp:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
       
        g = torch.Generator().manual_seed(seed + fold_idx)
       
        train_loader = DataLoader(
            train_dataset,
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
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
       
        yield fold_idx, train_loader, val_loader
