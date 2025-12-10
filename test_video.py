import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv
from data.video_data import create_uadfv_dataloaders

# تابع کمکی برای تنظیم seed (اگر ندارید، می‌توانید آن را حذف کنید)
def set_global_seed(seed: int):
    # این تابع را باید خودتان پیاده‌سازی کنید یا از کتابخانه مناسب وارد کنید
    # برای مثال:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to {seed}")


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
        self.num_frames = getattr(args, 'num_frames', 32)
        self.image_size = 256
        self.teacher_params = 23.51  # میلیون پارامتر
        self.teacher_video_flops = 170.59  # GFLOPs برای 32 فریم

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")

    def dataload(self):
        print(f"==> Loading {self.dataset_mode} test dataset..")
        if self.dataset_mode == 'uadfv':
            # اصلاح: فقط لودر تست را استخراج می‌کنیم
            _, _, self.test_loader = create_uadfv_dataloaders(
                root_dir=self.dataset_dir,
                num_frames=self.num_frames,
                image_size=self.image_size,
                train_batch_size=1, # مهم: سایز آموزشی در اینجا اهمیتی ندارد
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
        self.student.ticket = True  # فعال کردن حالت نهایی pruning

        # --- بخش اصلاح‌شده: محاسبه دقیق FLOPs و پارامترهای مؤثر ---
        print("\n" + "="*80)
        print("                    محاسبه معیارهای عملکردی")
        print("="*80)

        # محاسبه صحیح FLOPs با استفاده از متد خود مدل دانشجو
        student_flops = self.student.get_video_flops_sampled(num_sampled_frames=self.num_frames) / 1e9  # GFLOPs
        
        # محاسبه تعداد پارامترهای مؤثر (غیرمهر و موم شده)
        effective_params = 0
        for name, module in self.student.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    effective_params += torch.sum(module.weight_mask).item()
                else:
                    effective_params += module.weight.nelement()
                
                if module.bias is not None:
                    if hasattr(module, 'bias_mask'):
                        effective_params += torch.sum(module.bias_mask).item()
                    else:
                        effective_params += module.bias.nelement()
        
        student_params = effective_params / 1e6  # MParams

        # --- بخش اصلاح‌شده: گزارش نتایج مقایسه‌ای با مدل معلم ---
        print("\n" + "="*80)
        print("          مقایسه مدل دانشجو با مدل معلم (Teacher)")
        print("="*80)
        print(f"مدل معلم (Teacher)          : ۳۲ فریم  →  {self.teacher_video_flops:6.2f} GFLOPs | {self.teacher_params:5.2f}M params")
        print(f"مدل دانشجو (Student)        {self.num_frames:2d} فریم  →  {student_flops:6.2f} GFLOPs | {student_params:5.2f}M params")
        print("-"*80)
        
        # محاسبه درصد کاهش
        flops_reduction = (self.teacher_video_flops - student_flops) / self.teacher_video_flops * 100
        params_reduction = (self.teacher_params - student_params) / self.teacher_params * 100
        
        print(f"               کاهش FLOPs: {flops_reduction:6.2f}%")
        print(f"           کاهش پارامترها: {params_reduction:6.2f}%")
        print("="*80)
        
        # --- بخش تست دقت ---
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), ncols=100, desc="Testing") as _tqdm:
                for videos, targets in self.test_loader:
                    videos = videos.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True).float()
                    
                    batch_size, num_frames, C, H, W = videos.shape
                    videos_flat = videos.view(-1, C, H, W)

                    logits_student, _ = self.student(videos_flat)
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
        # ⚠️ مسیر فایل مدل خود را اینجا به درستی وارد کنید ⚠️
        self.sparsed_student_ckpt_path = '/kaggle/working/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt'
        self.num_frames = 32

if __name__ == '__main__':
    set_global_seed(42)
    args = Args()
    
    if not os.path.exists(args.sparsed_student_ckpt_path):
        print(f"ERROR: Student checkpoint not found at '{args.sparsed_student_ckpt_path}'")
        print("Please update the 'sparsed_student_ckpt_path' in the Args class.")
    else:
        test_pipeline = Test(args)
        test_pipeline.main()
