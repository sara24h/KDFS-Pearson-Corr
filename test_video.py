import os
import torch
import torch.nn as nn
from tqdm import tqdm
from thop import profile
from model.teacher.ResNet import ResNet_50_hardfakevsreal  # مسیر دقیق معلم خودت
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv

class Test:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.num_frames = getattr(args, 'num_frames', 16)

        # عدد دقیق معلم شما از قبل (ثابت و معتبر)
        self.teacher_video_flops = 170.59  # GFLOPs برای 32 فریم

    def build_models(self):
        print("==> Loading Teacher Model (for FLOPs reference)...")
        teacher = ResNet_50_hardfakevsreal()
        teacher.fc = nn.Linear(teacher.fc.in_features, 1)
        teacher.to(self.device)
        teacher.eval()

        print("==> Loading Pruned Student Model...")
        student = ResNet_50_sparse_uadfv()
        student.dataset_type = "uadfv"

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
        student.ticket = True  # مهم: پرونینگ فعال

        return teacher, student

    def test(self):
        teacher, student = self.build_models()

        # ورودی یک فریم (256x256)
        dummy = torch.randn(1, 3, 256, 256).to(self.device)

        # FLOPs یک فریم از دانشجو
        flops_student_frame, params_student = profile(student, inputs=(dummy,), verbose=False)
        flops_student_video = flops_student_frame * self.num_frames / 1e9  # GFLOPs

        print("\n" + "="*80)
        print("          مقایسه دقیق با مدل معلم شما (170.59 GFLOPs)")
        print("="*80)
        print(f"مدل معلم (Teacher)          : ۳۲ فریم  →  ۱۷۰٫۵۹ GFLOPs")
        print(f"مدل دانشجو (Student)        {self.num_frames:2d} فریم  →  {flops_student_video:6.2f} GFLOPs")
        print("-"*80)
        reduction = (self.teacher_video_flops - flops_student_video) / self.teacher_video_flops * 100
        print(f"               کاهش FLOPs: {reduction:6.2f}%")
        print(f"           کاهش پارامترها: {(23.51 - params_student/1e6)/23.51*100:6.2f}%")
        print("="*80)

        # تست دقت روی مجموعه تست
        from data.video_data import create_uadfv_dataloaders
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
        with torch.no_grad()
        with tqdm(test_loader, desc="Testing Accuracy") as pbar:
            for videos, labels in pbar:
                videos = videos.to(self.device)
                labels = labels.to(self.device).float()

                B, T, C, H, W = videos.shape
                videos = videos.view(-1, C, H, W)
                logits, _ = student(videos)
                logits = logits.view(B, T).mean(dim=1)
                preds = (torch.sigmoid(logits) > 0.5).float()

                correct += (preds == labels).sum().item()
                total += B
                pbar.set_postfix(Acc=f"{100.*correct/total:.2f}%")

        accuracy = 100. * correct / total
        print(f"\nدقت نهایی روی مجموعه تست (با {self.num_frames} فریم): {accuracy:.2f}%")

# استفاده
class Args:
    dataset_dir = "/kaggle/input/uadfv-dataset/UADFV"
    device = "cuda"
    test_batch_size = 8
    sparsed_student_ckpt_path = "/kaggle/working/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt"  # مسیر خودت رو بذار

if __name__ == '__main__':
    args = Args()
    tester = Test(args)
    tester.test()
