import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv
from data.video_data import create_uadfv_dataloaders


class Test:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.num_frames = getattr(args, 'num_frames', 16)
        self.teacher_num_frames = getattr(args, 'teacher_num_frames', 32)

    def build_models(self):
        print("بارگذاری مدل معلم (فقط برای مقایسه)...")
        teacher = ResNet_50_hardfakevsreal()
        teacher.fc = nn.Linear(teacher.fc.in_features, 1)
        teacher.to(self.device)
        teacher.eval()

        print("بارگذاری مدل دانشجوی هرس‌شده...")
        student = ResNet_50_sparse_uadfv()
        student.dataset_type = "uadfv"

        ckpt = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("student", ckpt)

        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict

        student.load_state_dict(state_dict, strict=True)
        student.to(self.device)
        student.eval()
        student.ticket = True  # پرونینگ فعال
        
        return teacher, student

    def calculate_flops(self, model, num_frames):
        
        if hasattr(model, 'get_video_flops_sampled'):
            # برای student که متد جدید دارد
            flops = model.get_video_flops_sampled(num_sampled_frames=num_frames)
            return flops / 1e9  # تبدیل به GFLOPs
        else:
          
            flops_per_frame = 5.39  # GFLOPs
            return flops_per_frame * num_frames

    def test(self):
        teacher, student = self.build_models()

        print("\n" + "="*85)
        print("محاسبه FLOPs...")
        print("-"*85)
        
        # Teacher FLOPs
        teacher_flops = self.calculate_flops(teacher, self.teacher_num_frames)
        print(f"Teacher ({self.teacher_num_frames} frames): {teacher_flops:.2f} GFLOPs")
        
        # Student FLOPs
        student_flops = self.calculate_flops(student, self.num_frames)
        print(f"Student ({self.num_frames} frames): {student_flops:.2f} GFLOPs")
        
        # محاسبه پارامترها
        teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
        student_params = sum(p.numel() for p in student.parameters()) / 1e6
        
        print("\n" + "="*85)
        print("                      مقایسه مدل‌های Teacher و Student")
        print("="*85)
        print(f"Teacher: {self.teacher_num_frames:2d} frames → {teacher_flops:7.2f} GFLOPs | {teacher_params:5.2f}M params")
        print(f"Student: {self.num_frames:2d} frames → {student_flops:7.2f} GFLOPs | {student_params:5.2f}M params")
        print("-"*85)
        
        # کاهش FLOPs
        flops_reduction = (teacher_flops - student_flops) / teacher_flops * 100
        params_reduction = (teacher_params - student_params) / teacher_params * 100
        
        print(f"FLOPs Reduction  : {flops_reduction:6.2f}%")
        print(f"Params Reduction : {params_reduction:6.2f}%")
        print("="*85)

        # تست دقت
        print("\nبارگذاری داده‌های تست...")
        _, _, test_loader = create_uadfv_dataloaders(
            root_dir=self.args.dataset_dir,
            num_frames=self.num_frames,
            image_size=256,
            train_batch_size=1,
            eval_batch_size=self.test_batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )

        correct = 0
        total = 0
        
        print(f"\nتست دقت روی {len(test_loader)} batch...")
        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc="Testing", ncols=80):
                videos = videos.to(self.device)
                labels = labels.to(self.device).float()

                B, T, C, H, W = videos.shape
                videos = videos.view(-1, C, H, W)

                logits, _ = student(videos)
                logits = logits.view(B, T).mean(dim=1)
                preds = (torch.sigmoid(logits) > 0.5).float()

                correct += (preds == labels).sum().item()
                total += B

        accuracy = 100.0 * correct / total
        
        print("\n" + "="*85)

        print(f"Test Accuracy    : {accuracy:.2f}%")
        print(f"FLOPs Reduction  : {flops_reduction:.2f}%")
        print(f"Student FLOPs    : {student_flops:.2f} GFLOPs (با {self.num_frames} فریم)")
        print("="*85)

    def main(self):
        self.test()


# ──────── تنظیمات ────────
class Args:
    def __init__(self):
        self.dataset_dir = "/kaggle/input/uadfv-dataset/UADFV"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_batch_size = 8
        self.sparsed_student_ckpt_path = "/kaggle/working/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt"
        self.num_frames = 16
        self.teacher_num_frames = 32  # ⭐ تعداد فریم‌های teacher


if __name__ == '__main__':
    args = Args()
    tester = Test(args)
    tester.main()
