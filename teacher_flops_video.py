import torch
import torch.nn as nn
from thop import profile
from ptflops import get_model_complexity_info
from torchvision.models import resnet50
from model.teacher.ResNet import ResNet_50_hardfakevsreal  # Assuming this is available in your environment
import cv2
import numpy as np
import os
import random
from pathlib import Path
from torchvision import transforms
import pandas as pd  # For formatting label distributions like in the example

# Install required packages if needed (comment out if already installed)
# !pip install thop ptflops

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

class UADFVDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_size=256,
                 transform=None, sampling_strategy='uniform',
                 split='train', split_ratio=(0.7, 0.15, 0.15), seed=42):
       
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.sampling_strategy = sampling_strategy  # Not used since we load all frames
        self.split = split
        self.seed = seed
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
            else: # val / test
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        # بارگذاری و تقسیم ویدیوها
        self.video_list = self._load_and_split(split_ratio)
        print(f"[{split.upper()}] {len(self.video_list)} videos loaded.")

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
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])
        if self.split == 'train':
            return video_list[:train_end]
        elif self.split == 'val':
            return video_list[train_end:val_end]
        elif self.split == 'test':
            return video_list[val_end:]
        else:
            raise ValueError("split must be train/val/test")

    def sample_frames(self, total_frames: int):
        # Modified to use all frames
        return list(range(total_frames))

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
            frames = torch.zeros(1, 3, self.image_size, self.image_size)  # Minimal placeholder
        finally:
            random.setstate(r_state)
            np.random.set_state(np_state)
        return frames, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    frames_list, labels = zip(*batch)
    max_t = max(f.size(0) for f in frames_list)
    padded_frames = []
    for f in frames_list:
        pad = (0, 0, 0, 0, 0, 0, 0, max_t - f.size(0))  # Pad along T (dim=0), format for 4D: (pad_l W, pad_r W, pad_t H, pad_b H, pad_f C, pad_b C, pad_s T, pad_e T)
        padded = torch.nn.functional.pad(f, pad, mode='constant', value=0)
        padded_frames.append(padded)
    return torch.stack(padded_frames), torch.stack(labels)

def create_uadfv_dataloaders(
    root_dir,
    image_size=256,
    train_batch_size=4,  # Smaller batch due to variable and potentially large T
    eval_batch_size=8,
    num_workers=4,
    pin_memory=True,
    ddp=False,
    sampling_strategy='uniform',  # Ignored
    seed=42
):
    train_ds = UADFVDataset(root_dir, image_size,
                            sampling_strategy=sampling_strategy,
                            split='train', seed=seed)
    val_ds = UADFVDataset(root_dir, image_size,
                          sampling_strategy=sampling_strategy,
                          split='val', seed=seed)
    test_ds = UADFVDataset(root_dir, image_size,
                           sampling_strategy=sampling_strategy,
                           split='test', seed=seed)
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, shuffle=False)
        shuffle = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle = True
    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=train_batch_size,
                                               shuffle=shuffle,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory,
                                               drop_last=True,
                                               worker_init_fn=worker_init_fn,
                                               generator=g,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=eval_batch_size,
                                             shuffle=False,
                                             sampler=val_sampler,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=eval_batch_size,
                                              shuffle=False,
                                              sampler=test_sampler,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              worker_init_fn=worker_init_fn,
                                              collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

def print_dataset_stats(ds, split_name):
    print(f"Sample {split_name} video paths:")
    for i in range(min(5, len(ds.video_list))):
        path, label = ds.video_list[i]
        label_str = 'Fake' if label == 0 else 'Real'
        print(i, f"{split_name.capitalize()}/{label_str}/{os.path.basename(path)}")
    print(f"Total {split_name} dataset size: {len(ds)}")
    labels = [label for _, label in ds.video_list]
    label_df = pd.DataFrame({'label': ['Fake' if l == 0 else 'Real' for l in labels]})
    print(f"{split_name.capitalize()} label distribution:\n{label_df['label'].value_counts().to_string()}")

def calculate_avg_frames(all_ds):
    frame_counts = []
    for ds in all_ds:
        for path, _ in ds.video_list:
            cap = cv2.VideoCapture(path)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_counts.append(count)
    avg_frames = sum(frame_counts) / len(frame_counts) if frame_counts else 0
    return avg_frames

def calculate_flops_params_for_video(model_path, avg_frames, image_size=256):
    print(f"Loading model from: {model_path}")
    model = ResNet_50_hardfakevsreal()
   
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'student' in checkpoint:
             state_dict = checkpoint['student']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # حذف 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
           
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
       
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Please ensure the model architecture in 'get_model()' matches the checkpoint.")
        return
    model.eval() # تنظیم مدل در حالت ارزیابی
    # 3. ایجاد ورودی جعلی برای thop
    # ورودی برای ResNet50 باید یک فریم باشد
    input_tensor = torch.randn(1, 3, image_size, image_size)
    # 4. محاسبه FLOPs و پارامترها برای یک فریم با thop
    flops_per_frame_thop, params_thop = profile(model, inputs=(input_tensor,), verbose=False)
    # 5. محاسبه FLOPs برای کل ویدیو با thop
    flops_per_video_thop = flops_per_frame_thop * avg_frames
    # ptflops
    flops_ptflops, params_ptflops = get_model_complexity_info(model, (3, image_size, image_size), as_strings=True,
                                                              print_per_layer_stat=True, verbose=True)
    print(flops_ptflops)
    print(params_ptflops)
    print(f"resnet50 MACs (thop): {flops_per_video_thop}")
    print(f"resnet50 Parameters (thop): {params_thop}")
    # 6. نمایش نتایج
    print("\n" + "="*70)
    print("Calculation Results for Video Model")
    print("="*70)
    print(f"Model Configuration:")
    print(f" - Architecture: ResNet50 (for binary classification)")
    print(f" - Input Frame Size: {image_size}x{image_size}")
    print(f" - Average Number of Frames per Video: {int(avg_frames)}")
    print("-" * 70)
    print(f"Model Metrics:")
    print(f" - Total Parameters: {params_thop/1e6:.2f} M")
    print(f" - FLOPs per Frame: {flops_per_frame_thop/1e9:.2f} GFLOPs")
    print(f" - FLOPs per Video: {flops_per_video_thop/1e9:.2f} GFLOPs")
    print("="*70)

if __name__ == "__main__":
    root_dir = "/kaggle/input/uadfv-dataset/UADFV"
    TEACHER_MODEL_PATH = "/kaggle/input/10k_teacher_beaet/pytorch/default/1/10k-teacher_model_best.pth"
    IMAGE_SIZE = 256

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = create_uadfv_dataloaders(
        root_dir=root_dir,
        image_size=IMAGE_SIZE,
        train_batch_size=4,
        eval_batch_size=8,
        num_workers=4,
        pin_memory=True,
        ddp=False,
        sampling_strategy='uniform'  # Ignored
    )

    print("UADFV dataset statistics:")
    print_dataset_stats(train_ds, 'train')
    print_dataset_stats(val_ds, 'validation')
    print_dataset_stats(test_ds, 'test')

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Validation loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    # Sample batches
    for videos, labels in train_loader:
        print("Sample train batch video shape:", videos.shape)  # [B, max_T, C, H, W]
        print("Sample train batch labels:", labels.tolist())
        break

    for videos, labels in val_loader:
        print("Sample validation batch video shape:", videos.shape)
        print("Sample validation batch labels:", labels.tolist())
        break

    for videos, labels in test_loader:
        print("Sample test batch video shape:", videos.shape)
        print("Sample test batch labels:", labels.tolist())
        break

    # Calculate average frames
    all_ds = [train_ds, val_ds, test_ds]
    avg_frames = calculate_avg_frames(all_ds)
    print("Average frames per video:", avg_frames)

    # Calculate FLOPs and params
    calculate_flops_params_for_video(
        model_path=TEACHER_MODEL_PATH,
        avg_frames=avg_frames,
        image_size=IMAGE_SIZE
    )
