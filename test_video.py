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
        
        print(f"\nبارگذاری checkpoint از: {self.sparsed_student_ckpt_path}")
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
        
        # بررسی محتوای checkpoint
        print("کلیدهای موجود در checkpoint:")
        for key in ckpt_student.keys():
            if isinstance(ckpt_student[key], dict):
                print(f"  - {key}: {len(ckpt_student[key])} آیتم")
            else:
                print(f"  - {key}: {type(ckpt_student[key])}")
        
        state_dict = ckpt_student.get("student", ckpt_student)
        
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '', 1)
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        # بررسی وجود ماسک‌ها در state_dict
        mask_count = sum(1 for k in state_dict.keys() if 'mask' in k)
        print(f"\nتعداد ماسک‌ها در checkpoint: {mask_count}")
        if mask_count == 0:
            print("⚠️ هشدار: هیچ ماسکی در checkpoint یافت نشد!")
            print("   مدل احتمالاً prune نشده است.")
        
        self.student.load_state_dict(state_dict, strict=True)
        self.student.to(self.device)
        print(f"Model loaded on {self.device}")

    def analyze_pruning_status(self):
        """بررسی دقیق وضعیت pruning مدل"""
        
        print("\n" + "="*80)
        print("تحلیل وضعیت Pruning")
        print("="*80)
        
        has_masks = False
        mask_layers = []
        
        for name, module in self.student.named_modules():
            if hasattr(module, 'weight_mask'):
                has_masks = True
                mask = module.weight_mask
                total = mask.numel()
                active = torch.sum(mask).item()
                sparsity = (1 - active/total) * 100
                
                mask_layers.append({
                    'name': name,
                    'total': total,
                    'active': active,
                    'sparsity': sparsity
                })
        
        if not has_masks:
            print("❌ مدل فاقد ماسک‌های pruning است!")
            print("   احتمالات:")
            print("   1. Checkpoint قبل از pruning بوده است")
            print("   2. ماسک‌ها در checkpoint ذخیره نشده‌اند")
            print("   3. تنظیم 'ticket=True' کافی نیست - نیاز به apply_mask() دارید")
            return False
        
        print(f"✓ تعداد لایه‌های دارای ماسک: {len(mask_layers)}")
        print("\nجزئیات pruning هر لایه:")
        print("-"*80)
        
        total_weights = 0
        total_active = 0
        
        for layer in mask_layers[:10]:  # فقط 10 لایه اول را نمایش می‌دهیم
            print(f"{layer['name']:40s} | Sparsity: {layer['sparsity']:6.2f}% | "
                  f"Active: {layer['active']:8d}/{layer['total']:8d}")
            total_weights += layer['total']
            total_active += layer['active']
        
        if len(mask_layers) > 10:
            print(f"... و {len(mask_layers)-10} لایه دیگر")
            for layer in mask_layers[10:]:
                total_weights += layer['total']
                total_active += layer['active']
        
        overall_sparsity = (1 - total_active/total_weights) * 100
        print("-"*80)
        print(f"میانگین کلی Sparsity: {overall_sparsity:.2f}%")
        print("="*80)
        
        return True

    def calculate_model_metrics(self):
        """محاسبه دقیق پارامترها و FLOPs"""
        
        # محاسبه پارامترها
        total_params = sum(p.numel() for p in self.student.parameters())
        
        # محاسبه پارامترهای فعال
        effective_params = 0
        for name, module in self.student.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    effective_params += torch.sum(module.weight_mask).item()
                else:
                    effective_params += module.weight.numel()
                
                if module.bias is not None:
                    if hasattr(module, 'bias_mask'):
                        effective_params += torch.sum(module.bias_mask).item()
                    else:
                        effective_params += module.bias.numel()
        
        # اضافه کردن پارامترهای BatchNorm و غیره
        for name, param in self.student.named_parameters():
            module_name = '.'.join(name.split('.')[:-1])
            try:
                module = self.student.get_submodule(module_name) if module_name else self.student
                if not isinstance(module, (nn.Conv2d, nn.Linear)):
                    effective_params += param.numel()
            except:
                effective_params += param.numel()
        
        sparsity = (total_params - effective_params) / total_params * 100 if total_params > 0 else 0
        
        print("\n" + "="*80)
        print("محاسبه پارامترها:")
        print("-"*80)
        print(f"کل پارامترها:         {total_params/1e6:8.2f} M")
        print(f"پارامترهای فعال:       {effective_params/1e6:8.2f} M")
        print(f"پارامترهای حذف شده:    {(total_params-effective_params)/1e6:8.2f} M")
        print(f"نرخ Sparsity:          {sparsity:8.2f} %")
        print("="*80)
        
        # محاسبه FLOPs
        print("\nمحاسبه FLOPs...")
        
        # استفاده از متد داخلی
        try:
            # ⚠️ IMPORTANT: این متد احتمالاً FLOPs یک فریم sample شده را برمی‌گرداند
            # نه کل 32 فریم!
            student_flops_single = self.student.get_video_flops_sampled(
                num_sampled_frames=self.num_frames
            ) / 1e9
            
            print(f"⚠️ توجه: متد get_video_flops_sampled احتمالاً فقط یک فریم sample شده")
            print(f"   را حساب می‌کند نه کل {self.num_frames} فریم!")
            print(f"   FLOPs گزارش شده: {student_flops_single:.2f} GFLOPs")
            
            # تخمین FLOPs واقعی برای تمام فریم‌ها
            # فرض: اگر متد فقط 1 فریم را حساب کرده، باید در تعداد فریم‌ها ضرب شود
            estimated_total_flops = student_flops_single * self.num_frames
            print(f"   تخمین FLOPs کل ({self.num_frames} فریم): {estimated_total_flops:.2f} GFLOPs")
            
            # استفاده از عدد کمتر برای مقایسه منصفانه
            student_flops = student_flops_single
            
        except Exception as e:
            print(f"خطا در محاسبه FLOPs: {e}")
            student_flops = 0.0
        
        return {
            'total_params': total_params / 1e6,
            'effective_params': effective_params / 1e6,
            'pruned_params': (total_params - effective_params) / 1e6,
            'sparsity': sparsity,
            'student_flops': student_flops,
            'estimated_total_flops': estimated_total_flops if 'estimated_total_flops' in locals() else student_flops
        }

    def test(self):
        self.student.eval()
        
        # 🔧 FIX 1: بررسی و فعال‌سازی صحیح pruning
        print("\n" + "="*80)
        print("فعال‌سازی حالت Pruning")
        print("="*80)
        
        # روش 1: تنظیم ticket
        self.student.ticket = True
        print("✓ ticket = True")
        
        # روش 2: اگر متد apply_mask وجود دارد، آن را فراخوانی کنید
        if hasattr(self.student, 'apply_mask'):
            self.student.apply_mask()
            print("✓ apply_mask() فراخوانی شد")
        
        # روش 3: بررسی وجود متد get_sparse_model
        if hasattr(self.student, 'get_sparse_model'):
            print("⚠️ توجه: متد get_sparse_model() وجود دارد - ممکن است نیاز باشد فراخوانی شود")
        
        # تحلیل وضعیت pruning
        has_pruning = self.analyze_pruning_status()
        
        if not has_pruning:
            print("\n" + "="*80)
            print("⚠️ هشدار مهم: مدل prune نشده است!")
            print("="*80)
            print("راه‌حل‌های پیشنهادی:")
            print("1. مطمئن شوید checkpoint صحیح است (بعد از pruning)")
            print("2. بررسی کنید که آیا باید از checkpoint دیگری استفاده کنید")
            print("3. ممکن است نیاز باشد script pruning را اجرا کنید")
            print("="*80 + "\n")
        
        # محاسبه معیارها
        metrics = self.calculate_model_metrics()
        
        # گزارش مقایسه‌ای
        print("\n" + "="*80)
        print("          مقایسه با مدل معلم (Teacher)")
        print("="*80)
        print(f"مدل معلم (Teacher):")
        print(f"  - FLOPs (32 فریم):     {self.teacher_video_flops:8.2f} GFLOPs")
        print(f"  - پارامترها:            {self.teacher_params:8.2f} M")
        print("-"*80)
        print(f"مدل دانشجو (Student):")
        print(f"  - FLOPs (گزارش شده):    {metrics['student_flops']:8.2f} GFLOPs")
        if 'estimated_total_flops' in metrics:
            print(f"  - FLOPs (تخمینی کل):    {metrics['estimated_total_flops']:8.2f} GFLOPs")
        print(f"  - پارامترها فعال:       {metrics['effective_params']:8.2f} M")
        print(f"  - Sparsity:             {metrics['sparsity']:8.2f} %")
        print("-"*80)
        
        # محاسبه کاهش (با FLOPs تخمینی)
        flops_for_comparison = metrics.get('estimated_total_flops', metrics['student_flops'])
        flops_reduction = ((self.teacher_video_flops - flops_for_comparison) / 
                          self.teacher_video_flops * 100)
        params_reduction = ((self.teacher_params - metrics['effective_params']) / 
                           self.teacher_params * 100)
        
        print(f"کاهش FLOPs:     {flops_reduction:7.2f} %")
        print(f"کاهش پارامترها: {params_reduction:7.2f} %")
        print("="*80)
        
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
        print(f"  ✓ Accuracy:         {final_acc:.2f}%")
        print(f"  ✓ کاهش FLOPs:       {flops_reduction:.2f}%")
        print(f"  ✓ کاهش پارامترها:   {params_reduction:.2f}%")
        print(f"  ✓ Sparsity واقعی:   {metrics['sparsity']:.2f}%")
        
        if metrics['sparsity'] < 1.0:
            print("\n  ⚠️ توجه: Sparsity نزدیک به صفر است - مدل prune نشده!")
        
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
