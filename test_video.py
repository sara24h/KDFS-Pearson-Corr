import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import random
from torchvision.models import resnet50 # اضافه شده
from thop import profile 
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv

# --- تنظیم اولیه برای تکرارپذیری نتایج ---
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

# --- کلاس دیتاست UADFV ---
# (این کد همان کد ارائه شده توسط شماست، بدون تغییر)
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
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:  # val / test
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        self.video_list = self._load_and_split(split_ratio)
        print(f"[{split.upper()}] {len(self.video_list)} videos loaded.")

    def _load_and_split(self, split_ratio):
        video_list = []
        for p in sorted((self.root_dir / 'fake').glob('*.mp4')):
            if not p.name.startswith('.'): video_list.append((str(p), 0))
        for p in sorted((self.root_dir / 'real').glob('*.mp4')):
            if not p.name.startswith('.'): video_list.append((str(p), 1))
        
        rng = random.Random(self.seed)
        rng.shuffle(video_list)
        total = len(video_list)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])

        if self.split == 'train': return video_list[:train_end]
        elif self.split == 'val': return video_list[train_end:val_end]
        elif self.split == 'test': return video_list[val_end:]
        else: raise ValueError("split must be train/val/test")

    def sample_frames(self, total_frames: int):
        if total_frames <= self.num_frames:
            idxs = np.random.choice(total_frames, self.num_frames, replace=True)
            return sorted(idxs.tolist())
        if self.sampling_strategy == 'uniform':
            return np.linspace(0, total_frames-1, self.num_frames, dtype=int).tolist()
        elif self.sampling_strategy == 'random':
            idxs = np.random.choice(total_frames, self.num_frames, replace=False)
            return sorted(idxs.tolist())
        else: raise ValueError("sampling_strategy: uniform / random")

    def load_video(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise IOError(f"Cannot open {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self.sample_frames(total)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try: frame = self.transform(frame)
                except Exception as e:
                    print(f"Transform error on frame from {path}: {e}")
                    frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(frame)
            else:
                fallback = frames[-1].clone() if frames else torch.zeros(3, self.image_size, self.image_size)
                frames.append(fallback)
        cap.release()
        return torch.stack(frames)

    def __len__(self): return len(self.video_list)
    def __getitem__(self, idx):
        path, label = self.video_list[idx]
        frames = self.load_video(path)
        return frames, torch.tensor(label, dtype=torch.float32)

def create_uadfv_dataloaders(root_dir, num_frames=16, image_size=256, train_batch_size=8, eval_batch_size=16, num_workers=4, pin_memory=True, ddp=False, sampling_strategy='uniform', seed=42):
    train_ds = UADFVDataset(root_dir, num_frames, image_size, sampling_strategy=sampling_strategy, split='train', seed=seed)
    val_ds   = UADFVDataset(root_dir, num_frames, image_size, sampling_strategy=sampling_strategy, split='val', seed=seed)
    test_ds  = UADFVDataset(root_dir, num_frames, image_size, sampling_strategy=sampling_strategy, split='test', seed=seed)
    sampler = None
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
    return test_loader

# --- کلاس تست اصلاح‌شده ---
class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.dataset_mode = args.dataset_mode
        self.num_frames = getattr(args, 'num_frames', 16)
        self.image_size = 256

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")

    def dataload(self):
        print(f"==> Loading {self.dataset_mode} test dataset..")
        if self.dataset_mode == 'uadfv':
            self.test_loader = create_uadfv_dataloaders(
                root_dir=self.dataset_dir,
                num_frames=self.num_frames,
                image_size=self.image_size,
                train_batch_size=self.test_batch_size,
                eval_batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                ddp=False,
                sampling_strategy='uniform'
            )
            print(f"{self.dataset_mode} test dataset loaded! Total batches: {len(self.test_loader)}")
        else:
            raise ValueError(f"This test script is currently configured only for 'uadfv' dataset mode.")

    def build_model(self):
        print("==> Building student model..")
        self.student = ResNet_50_sparse_uadfv()
        self.student.dataset_type = "uadfv" 
        
        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")
        
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt_student.get("student", ckpt_student)
        
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '', 1)
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.student.load_state_dict(state_dict, strict=True)
        self.student.to(self.device)
        print(f"Model loaded on {self.device}")

    def test(self):
        self.student.eval()
        self.student.ticket = True  # فعال کردن حالت نهایی پرuning

        # --- بخش جدید: محاسبه FLOPs و پارامترهای مدل دانشجو ---
        input_tensor = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        flops_pruned, params_pruned = profile(self.student, inputs=(input_tensor,), verbose=False)
        
        # --- بخش جدید: محاسبه FLOPs و پارامترهای مدل پایه (Baseline) ---
        print("\nCalculating baseline metrics for standard ResNet-50...")
        baseline_model = resnet50(pretrained=False)
        baseline_model.fc = nn.Linear(baseline_model.fc.in_features, 1) # تطبیق لایه نهایی
        baseline_model.to(self.device)
        flops_baseline, params_baseline = profile(baseline_model, inputs=(input_tensor,), verbose=False)
        
        # --- بخش جدید: محاسبه درصد کاهش ---
        params_reduction = (params_baseline - params_pruned) / params_baseline * 100
        flops_reduction = (flops_baseline - flops_pruned) / flops_baseline * 100
        
        # --- گزارش نهایی ---
        print("\n" + "="*70)
        print("Final Comparison Report")
        print("="*70)
        print(f"Baseline Model (Standard ResNet-50):")
        print(f"  - Total Parameters: {params_baseline/1e6:.2f} M")
        print(f"  - FLOPs per Frame: {flops_baseline/1e9:.2f} GFLOPs")
        print("-" * 70)
        print(f"Pruned Student Model:")
        print(f"  - Total Parameters: {params_pruned/1e6:.2f} M")
        print(f"  - FLOPs per Frame: {flops_pruned/1e9:.2f} GFLOPs")
        print("-" * 70)
        print(f"Reduction:")
        print(f"  - Params reduction: {params_reduction:.2f}%")
        print(f"  - FLOPs reduction: {flops_reduction:.2f}%")
        print("="*70)
        
        # --- بخش تست دقت ---
        # از آنجایی که meter را نداریم، یک ساده پیاده‌سازی می‌کنیم
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), ncols=100, desc="Testing") as _tqdm:
                for images, targets in self.test_loader:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True).float()
                    
                    batch_size, num_frames, C, H, W = images.shape
                    images = images.view(-1, C, H, W)

                    logits_student, _ = self.student(images)
                    logits_student = logits_student.squeeze(1)
                    logits_student = logits_student.view(batch_size, num_frames).mean(dim=1)
                    
                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

                    _tqdm.set_postfix(Acc=f"{(100.*correct/total):.2f}%")
                    _tqdm.update(1)

        print(f"\n[Test] Final Accuracy on {self.dataset_mode} dataset: {100.*correct/total:.2f}%")

    def main(self):
        print(f"Starting test pipeline for dataset: {self.dataset_mode}")
        self.dataload()
        self.build_model()
        self.test()

# --- کلاس کمکی برای نگهداری آرگومان‌ها ---
class Args:
    def __init__(self):
        self.dataset_mode = 'uadfv'
        self.dataset_dir = '/kaggle/input/uadfv-dataset/UADFV'
        self.num_workers = 4
        self.pin_memory = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_batch_size = 8
        self.sparsed_student_ckpt_path = '/path/to/your/resnet50_sparse_best.pt' # <-- مسیر فایل مدل خود را اینجا قرار دهید
        self.num_frames = 16

if __name__ == '__main__':
    set_global_seed(42)
    args = Args()
    
    if not os.path.exists(args.sparsed_student_ckpt_path):
        print(f"ERROR: Student checkpoint not found at '{args.sparsed_student_ckpt_path}'")
        print("Please update to 'sparsed_student_ckpt_path' in the Args class.")
    else:
        test_pipeline = Test(args)
        test_pipeline.main()
