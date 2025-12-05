import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
import numpy as np
import os
from torchvision import transforms
from pathlib import Path
import random
from typing import Tuple, List

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
    worker_seed = 42 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class UADFVDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, image_size=256,
                 transform=None, sampling_strategy='uniform',
                 split='train', split_ratio=(0.7, 0.15, 0.15), seed=42):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.sampling_strategy = sampling_strategy
        self.split = split
        self.seed = seed

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.video_list = self._load_video_paths(split_ratio)
        print(f"[{split.upper()}] Loaded {len(self.video_list)} videos")

    def _load_video_paths(self, split_ratio):
        video_list = []

        # Fake videos → label 0
        fake_dir = self.root_dir / 'fake'
        if fake_dir.exists():
            for p in sorted(fake_dir.glob('*.mp4')):
                if not p.name.startswith('.'):
                    video_list.append((str(p), 0))

        real_dir = self.root_dir / 'real'
        if real_dir.exists():
            for p in sorted(real_dir.glob('*.mp4')):
                if not p.name.startswith('.'):
                    video_list.append((str(p), 1))

        rng = random.Random(self.seed)
        rng.shuffle(video_list)

        total = len(video_list)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])

        if self.split == 'train':
            return video_list[:train_end]
        elif self.split == 'val':
            return video_list[train_end:val_end]
        elif self.split == 'test':
            return video_list[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def sample_frames(self, total_frames: int) -> List[int]:

        if total_frames < self.num_frames:

            indices = np.random.choice(total_frames, self.num_frames, replace=True)
            return sorted(indices)
        else:
            if self.sampling_strategy == 'uniform':
                return np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
            elif self.sampling_strategy == 'random':
                indices = np.random.choice(total_frames, self.num_frames, replace=False)
                return sorted(indices.tolist())
            elif self.sampling_strategy == 'first':
                return list(range(self.num_frames))
            else:
                raise ValueError(f"Unknown strategy: {self.sampling_strategy}")

    def load_video(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise IOError(f"Empty video: {video_path}")

        frame_indices = self.sample_frames(total_frames)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
            else:
                # fallback
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, self.image_size, self.image_size))

        cap.release()
        return torch.stack(frames)  # [T, C, H, W]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path, label = self.video_list[idx]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:

            per_sample_seed = self.seed + worker_info.id * 100000 + idx
        else:
            per_sample_seed = self.seed + idx

        rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        random.seed(per_sample_seed)
        np.random.seed(per_sample_seed)

        try:
            frames = self.load_video(video_path)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        finally:
       
            random.setstate(rng_state)
            np.random.set_state(np_rng_state)

        return frames, torch.tensor(label, dtype=torch.float32)

def create_uadfv_dataloaders(
    root_dir,
    num_frames=16,
    image_size=256,
    train_batch_size=8,
    eval_batch_size=16,
    num_workers=4,
    ddp=False,
    sampling_strategy='uniform',
    seed=42
):
    train_dataset = UADFVDataset(root_dir=root_dir, num_frames=num_frames, image_size=image_size,
                                 sampling_strategy=sampling_strategy, split='train', seed=seed)
    val_dataset   = UADFVDataset(root_dir=root_dir, num_frames=num_frames, image_size=image_size,
                                 sampling_strategy=sampling_strategy, split='val', seed=seed)
    test_dataset  = UADFVDataset(root_dir=root_dir, num_frames=num_frames, image_size=image_size,
                                 sampling_strategy=sampling_strategy, split='test', seed=seed)

    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset, shuffle=False)
        test_sampler  = DistributedSampler(test_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle = True

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    root_dir = "/kaggle/input/uadfv-dataset/UADFV" 

    train_loader, val_loader, test_loader = create_uadfv_dataloaders(
        root_dir=root_dir,
        num_frames=16,
        image_size=256,
        train_batch_size=4,
        eval_batch_size=8,
        num_workers=4,
        ddp=False,
        sampling_strategy='uniform' 
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    for videos, labels in train_loader:
        print("Batch shape :", videos.shape)      # [B, T, C, H, W]
        print("Labels shape:", labels.shape)
        print("Labels     :", labels.tolist())
        print("First video mean pixel:", videos[0].mean().item())
        break
        break
