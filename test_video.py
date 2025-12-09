import os
import torch
import torch.nn as nn
from tqdm import tqdm
# فرض می‌کنیم این فایل‌ها در مسیر صحیح پروژه شما قرار دارند
from model.teacher.ResNet import ResNet_50_hardfakevsreal  
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv
from data.video_data import create_uadfv_dataloaders

# =============================================================================
# کلاس اصلی برای تست و ارزیابی مدل
# =============================================================================
class Test:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.num_frames = getattr(args, 'num_frames', 32)
        self.image_size = 256

        # مقادیر ثابت و معتبر برای مدل معلم (Teacher)
        # این اعداد از قبل محاسبه شده‌اند و برای مقایسه استفاده می‌شوند
        self.teacher_params = 23.51  # میلیون پارامتر
        self.teacher_video_flops = 170.59  # GFLOPs برای 32 فریم

    def build_student_model(self):
        """
        فقط مدل دانشجو را بارگذاری می‌کند، چون مدل معلم فقط یک عدد ثابت برای مقایسه است.
        """
        print("==> Loading Pruned Student Model...")
        student = ResNet_50_sparse_uadfv()
        student.dataset_type = "uadfv"

        if not os.path.exists(self.sparsed_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.sparsed_student_ckpt_path}")

        ckpt = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("student", ckpt)
        
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '', 1)] = v
            state_dict = new_state_dict
        
        student.load_state_dict(state_dict, strict=True)
        student.to(self.device)
        student.eval()
        
        # مهم: فعال کردن حالت نهایی هرس کردن (pruning)
        student.ticket = True
        
        print("Student model loaded successfully.")
        return student

    def test(self):
        """
        تابع اصلی برای اجرای تست و محاسبه معیارها
        """
        student = self.build_student_model()

        # --- بخش ۱: محاسبه FLOPs و پارامترها ---
        print("\n" + "="*80)
        print("                    محاسبه معیارهای عملکردی")
        print("="*80)

        # محاسبه صحیح FLOPs با استفاده از متد خود مدل دانشجو
        # این متد ماسک‌های pruning را در نظر می‌گیرد
        student_flops = student.get_video_flops_sampled(num_sampled_frames=self.num_frames) / 1e9  # GFLOPs
        
        # محاسبه تعداد پارامترهای مدل دانشجو
        student_params = sum(p.numel() for p in student.parameters()) / 1e6  # MParams

        # --- بخش ۲: گزارش نتایج مقایسه‌ای ---
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

        # --- بخش ۳: تست دقت روی مجموعه داده تست ---
        print("\n==> Loading Test Dataset...")
        _, _, test_loader = create_uadfv_dataloaders(
            root_dir=self.args.dataset_dir,
            num_frames=self.num_frames,
            image_size=self.image_size,
            train_batch_size=1,
            eval_batch_size=self.test_batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )

        correct = 0
        total = 0
        
        print(f"\n==> Testing Accuracy on {len(test_loader)} batches...")
        # استفاده از with torch.no_grad() برای صرفه‌جویی در حافظه و افزایش سرعت
        with torch.no_grad():
            # استفاده از tqdm برای نمایش پیشرفت به زیبایی
            with tqdm(test_loader, desc="Testing Accuracy") as pbar:
                for videos, labels in pbar:
                    videos = videos.to(self.device)
                    labels = labels.to(self.device).float()

                    B, T, C, H, W = videos.shape
                    videos = videos.view(-1, C, H, W)

                    # پاس فوروارد مدل
                    logits, _ = student(videos)
                    logits = logits.view(B, T).mean(dim=1)
                    preds = (torch.sigmoid(logits) > 0.5).float()

                    correct += (preds == labels).sum().item()
                    total += B
                    pbar.set_postfix(Acc=f"{100.*correct/total:.2f}%")

        accuracy = 100. * correct / total
        print(f"\n{'='*80}")
        print(f"      دقت نهایی روی مجموعه تست (با {self.num_frames} فریم): {accuracy:.2f}%")
        print(f"{'='*80}")

    def main(self):
        self.test()

# =============================================================================
# کلاس برای نگهداری تنظیمات و آرگومان‌های ورودی
# =============================================================================
class Args:
    def __init__(self):
        # ⚠️ مسیر دیتاست و فایل مدل خود را اینجا وارد کنید ⚠️
        self.dataset_dir = "/kaggle/input/uadfv-dataset/UADFV"
        self.sparsed_student_ckpt_path = "/kaggle/working/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt"
        
        # تنظیمات دیگر
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_batch_size = 8
        self.num_frames = 32  # می‌توانید این عدد را به 32 تغییر دهید تا تأثیر آن را ببینید

# =============================================================================
# نقطه شروع اجرای اسکریپت
# =============================================================================
if __name__ == '__main__':
    args = Args()
    tester = Test(args)
    tester.main()
