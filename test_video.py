import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv
from data.video_data import create_uadfv_dataloaders

def set_global_seed(seed: int):
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

        # مقادیر مدل معلم (Teacher)
        self.teacher_params = 23.51  # میلیون پارامتر
        self.teacher_video_flops = 170.59  # GFLOPs برای 32 فریم

        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! Please check GPU setup.")

    def dataload(self):
        print(f"==> Loading {self.dataset_mode} test dataset..")
        if self.dataset_mode == 'uadfv':
            _, _, self.test_loader = create_uadfv_dataloaders(
                root_dir=self.dataset_dir,
                num_frames=self.num_frames,
                image_size=self.image_size,
                train_batch_size=1,
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

    def calculate_model_metrics(self):
        """محاسبه دقیق پارامترها و FLOPs"""
        
        # ✅ محاسبه صحیح پارامترها (شامل تمام لایه‌ها)
        total_params = 0
        effective_params = 0
        pruned_params = 0
        
        print("\n" + "="*80)
        print("جزئیات محاسبه پارامترها:")
        print("-"*80)
        
        for name, param in self.student.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            # پیدا کردن ماسک مرتبط
            mask = None
            if 'weight' in name:
                mask_name = name.replace('weight', 'weight_mask')
            elif 'bias' in name:
                mask_name = name.replace('bias', 'bias_mask')
            else:
                mask_name = None
            
            if mask_name:
                # جستجوی ماسک در مدل
                parts = mask_name.split('.')
                obj = self.student
                found_mask = True
                for part in parts:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        found_mask = False
                        break
                
                if found_mask and isinstance(obj, torch.Tensor):
                    mask = obj
                    effective_count = torch.sum(mask).item()
                    effective_params += effective_count
                    pruned_params += (param_count - effective_count)
                else:
                    # ماسک وجود ندارد - همه پارامترها فعال
                    effective_params += param_count
            else:
                # پارامترهای بدون ماسک (مثل BatchNorm)
                effective_params += param_count
        
        sparsity = (pruned_params / total_params) * 100 if total_params > 0 else 0
        
        print(f"کل پارامترها:         {total_params/1e6:8.2f} M")
        print(f"پارامترهای فعال:       {effective_params/1e6:8.2f} M")
        print(f"پارامترهای حذف شده:    {pruned_params/1e6:8.2f} M")
        print(f"نرخ Sparsity:          {sparsity:8.2f} %")
        print("="*80)
        
        # ✅ محاسبه FLOPs
        print("\nمحاسبه FLOPs...")
        
        # روش 1: استفاده از متد داخلی مدل
        try:
            student_flops_method1 = self.student.get_video_flops_sampled(
                num_sampled_frames=self.num_frames
            ) / 1e9
            print(f"FLOPs (متد داخلی):     {student_flops_method1:8.2f} GFLOPs")
        except Exception as e:
            print(f"خطا در محاسبه FLOPs با متد داخلی: {e}")
            student_flops_method1 = None
        
        # روش 2: محاسبه دستی برای یک فریم
        single_frame_flops = self._calculate_single_frame_flops()
        student_flops_method2 = (single_frame_flops * self.num_frames) / 1e9
        print(f"FLOPs (محاسبه دستی):   {student_flops_method2:8.2f} GFLOPs ({self.num_frames} فریم)")
        
        # انتخاب FLOPs نهایی
        student_flops = student_flops_method1 if student_flops_method1 is not None else student_flops_method2
        
        return {
            'total_params': total_params / 1e6,
            'effective_params': effective_params / 1e6,
            'pruned_params': pruned_params / 1e6,
            'sparsity': sparsity,
            'student_flops': student_flops
        }
    
    def _calculate_single_frame_flops(self):
        """محاسبه FLOPs برای یک فریم (تقریبی)"""
        total_flops = 0
        
        for name, module in self.student.named_modules():
            if isinstance(module, nn.Conv2d):
                # FLOPs = 2 × Cin × Cout × K × K × H × W
                # (فاکتور 2 برای ضرب و جمع)
                
                # بررسی وجود ماسک
                if hasattr(module, 'weight_mask'):
                    # تعداد کانال‌های ورودی/خروجی فعال
                    active_weights = torch.sum(module.weight_mask).item()
                    # تقریب: استفاده از نسبت ماسک
                    mask_ratio = active_weights / module.weight.numel()
                else:
                    mask_ratio = 1.0
                
                # محاسبه FLOPs (فرض: ورودی 256x256)
                # توجه: این یک تقریب است و باید با توجه به معماری دقیق‌تر شود
                k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                flops = 2 * module.in_channels * module.out_channels * k * k
                flops *= mask_ratio
                total_flops += flops * (256 * 256)  # فرض: spatial size اولیه
                
            elif isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    active_weights = torch.sum(module.weight_mask).item()
                else:
                    active_weights = module.weight.numel()
                
                total_flops += 2 * active_weights
        
        return total_flops

    def test(self):
        self.student.eval()
        self.student.ticket = True  # فعال کردن حالت نهایی pruning

        # ✅ محاسبه معیارها
        metrics = self.calculate_model_metrics()
        
        # ✅ گزارش مقایسه‌ای
        print("\n" + "="*80)
        print("          مقایسه مدل دانشجو با مدل معلم (Teacher)")
        print("="*80)
        print(f"مدل معلم (Teacher):")
        print(f"  - FLOPs:      {self.teacher_video_flops:8.2f} GFLOPs ({self.num_frames} فریم)")
        print(f"  - پارامترها: {self.teacher_params:8.2f} M")
        print("-"*80)
        print(f"مدل دانشجو (Student):")
        print(f"  - FLOPs:      {metrics['student_flops']:8.2f} GFLOPs ({self.num_frames} فریم)")
        print(f"  - پارامترها فعال: {metrics['effective_params']:8.2f} M")
        print(f"  - Sparsity:   {metrics['sparsity']:8.2f} %")
        print("-"*80)
        
        # محاسبه کاهش
        flops_reduction = ((self.teacher_video_flops - metrics['student_flops']) / 
                          self.teacher_video_flops * 100)
        params_reduction = ((self.teacher_params - metrics['effective_params']) / 
                           self.teacher_params * 100)
        
        print(f"کاهش FLOPs:     {flops_reduction:7.2f} %")
        print(f"کاهش پارامترها: {params_reduction:7.2f} %")
        print("="*80)
        
        # ⚠️ هشدارهای احتمالی
        if abs(metrics['sparsity'] - params_reduction) > 10:
            print("\n⚠️ هشدار: تفاوت قابل توجه بین Sparsity و کاهش پارامترها!")
            print("   این ممکن است نشان‌دهنده مشکل در محاسبه یا وجود پارامترهای")
            print("   غیر-prunable (مثل BatchNorm) باشد.\n")
        
        if flops_reduction > 90:
            print("\n⚠️ هشدار: کاهش FLOPs بیش از 90% غیرمعمول است!")
            print("   لطفاً متد get_video_flops_sampled() را بررسی کنید.")
            print("   ممکن است این متد frame sampling انجام دهد.\n")
        
        # تست دقت
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

        final_acc = 100. * correct / total
        print(f"\n[Test] Final Accuracy on {self.dataset_mode} dataset: {final_acc:.2f}%")
        
        # خلاصه نهایی
        print("\n" + "="*80)
        print("خلاصه نتایج:")
        print(f"  ✓ Accuracy: {final_acc:.2f}%")
        print(f"  ✓ کاهش FLOPs: {flops_reduction:.2f}%")
        print(f"  ✓ کاهش پارامترها: {params_reduction:.2f}%")
        print(f"  ✓ Sparsity: {metrics['sparsity']:.2f}%")
        print("="*80)

    def main(self):
        print(f"Starting test pipeline for dataset: {self.dataset_mode}")
        self.dataload()
        self.build_model()
        self.test()


class Args:
    def __init__(self):
        self.dataset_mode = 'uadfv'
        self.dataset_dir = '/kaggle/input/uadfv-dataset/UADFV'
        self.num_workers = 4
        self.pin_memory = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_batch_size = 8
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
