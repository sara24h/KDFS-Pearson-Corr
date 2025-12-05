import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
import numpy as np
import os
from torchvision import transforms
from pathlib import Path
import random


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
        
        # Load video paths
        self.video_list = self._load_video_paths(split_ratio)
        
        print(f"Loaded {len(self.video_list)} videos for {split} split")
    
    def _load_video_paths(self, split_ratio):
        """Load all video paths and their labels, then split"""
        video_list = []
        
        # Load fake videos
        fake_dir = self.root_dir / 'fake'
        if fake_dir.exists():
            fake_videos = sorted([f for f in fake_dir.glob('*.mp4') if not f.name.startswith('.')])
            for video_path in fake_videos:
                video_list.append((str(video_path), 1))  # Label 1 for fake
        
        # Load real videos
        real_dir = self.root_dir / 'real'
        if real_dir.exists():
            real_videos = sorted([f for f in real_dir.glob('*.mp4') if not f.name.startswith('.')])
            for video_path in real_videos:
                video_list.append((str(video_path), 0))  # Label 0 for real
        
        # Shuffle with fixed seed for reproducibility
        random.seed(self.seed)
        random.shuffle(video_list)
        
        # Split dataset
        total_videos = len(video_list)
        train_ratio, val_ratio, test_ratio = split_ratio
        
        train_end = int(total_videos * train_ratio)
        val_end = train_end + int(total_videos * val_ratio)
        
        if self.split == 'train':
            return video_list[:train_end]
        elif self.split == 'val':
            return video_list[train_end:val_end]
        elif self.split == 'test':
            return video_list[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def sample_frames(self, total_frames):
        """Sample frame indices based on strategy"""
        if total_frames < self.num_frames:
            # If video has fewer frames than needed, sample with replacement
            indices = np.random.choice(total_frames, self.num_frames, replace=True)
            indices = np.sort(indices)
        else:
            if self.sampling_strategy == 'uniform':
                # Sample uniformly across the video
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            elif self.sampling_strategy == 'random':
                # Random sampling
                indices = np.sort(np.random.choice(total_frames, self.num_frames, replace=False))
            elif self.sampling_strategy == 'first':
                # Take first N frames
                indices = np.arange(self.num_frames)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        return indices
    
    def load_video(self, video_path):
        """Load and process video frames"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise IOError(f"Video has 0 frames: {video_path}")
        
        frame_indices = self.sample_frames(total_frames)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply transforms
                frame = self.transform(frame)
                frames.append(frame)
            else:
                # If frame reading fails, use the last successful frame
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    # Create a black frame as fallback
                    black_frame = torch.zeros(3, self.image_size, self.image_size)
                    frames.append(black_frame)
        
        cap.release()
        
        # Stack frames: [num_frames, C, H, W]
        frames_tensor = torch.stack(frames)
        
        return frames_tensor
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video_path, label = self.video_list[idx]
        
        try:
            frames = self.load_video(video_path)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return zeros as fallback
            frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        
        return frames, torch.tensor(label, dtype=torch.float32)


def create_uadfv_dataloaders(root_dir, num_frames=16, image_size=256,
                             train_batch_size=8, eval_batch_size=16,
                             num_workers=4, pin_memory=True, ddp=False,
                             split_ratio=(0.7, 0.15, 0.15), seed=42,
                             sampling_strategy='uniform'):
  
    # Create datasets
    train_dataset = UADFVDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        image_size=image_size,
        sampling_strategy=sampling_strategy,
        split='train',
        split_ratio=split_ratio,
        seed=seed
    )
    
    val_dataset = UADFVDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        image_size=image_size,
        sampling_strategy=sampling_strategy,
        split='val',
        split_ratio=split_ratio,
        seed=seed
    )
    
    test_dataset = UADFVDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        image_size=image_size,
        sampling_strategy=sampling_strategy,
        split='test',
        split_ratio=split_ratio,
        seed=seed
    )
    
    # Create samplers for DDP
    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


# Test function
if __name__ == "__main__":
    # Example usage
    root_dir = "/path/to/UADFV"
    
    train_loader, val_loader, test_loader = create_uadfv_dataloaders(
        root_dir=root_dir,
        num_frames=16,
        image_size=256,
        train_batch_size=4,
        eval_batch_size=8,
        num_workers=4,
        ddp=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading one batch
    for videos, labels in train_loader:
        print(f"Video batch shape: {videos.shape}")  # [batch, num_frames, C, H, W]
        print(f"Labels shape: {labels.shape}")  # [batch]
        print(f"Sample labels: {labels}")
        break
