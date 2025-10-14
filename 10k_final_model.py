import torch
import os
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal


# 1. مسیر و لود فایل چک‌پوینت ورودی
input_save_path = '/kaggle/input/10k_pruned_model_resnet50/pytorch/default/1/resnet50_pruned_model_learnable_masks.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. مسیر خروجی برای ذخیره‌سازی مجدد (مهم: باید در /kaggle/working باشد)
output_save_path = '/kaggle/working/10k_final.pt'

# ----------------------------------------------------
# 3. لود، بازسازی و ذخیره‌سازی مدل
# ----------------------------------------------------

try:
    # الف) لود چک‌پوینت کامل ورودی
    checkpoint_loaded = torch.load(input_save_path, map_location=device)

    # ب) استخراج اطلاعات کلیدی
    model_state_dict = checkpoint_loaded['model_state_dict']
    masks = checkpoint_loaded['masks']
    
    # ج) ساخت مدل هرس‌شده با استفاده از ماسک‌ها
    model_pruned = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    # د) لود وزن‌های هرس‌شده
    model_pruned.load_state_dict(model_state_dict)
    
    model_pruned = model_pruned.to(device)
    model_pruned.eval()
    
    total_params = sum(p.numel() for p in model_pruned.parameters())
    print("✅ مدل هرس‌شده با موفقیت بازسازی و لود شد!")
    print(f"تعداد پارامترها: {total_params:,}")

    # ----------------------------------------------------
    # ه) ذخیره‌سازی مدل بازسازی‌شده
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("💾 ذخیره‌سازی مدل بازسازی‌شده...")
    print("="*70)
    
    # ذخیره مجدد تمام محتویات لازم برای بازسازی (وزن‌ها، ماسک‌ها، متاداده)
    checkpoint_to_save = {
        'model_state_dict': model_pruned.state_dict(),
        'masks': masks, # ذخیره ماسک‌ها برای بازسازی آسان در آینده
        'total_params': total_params,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal'
    }
    
    torch.save(checkpoint_to_save, output_save_path)
    
    # چاپ اطلاعات فایل ذخیره شده
    file_size_mb = os.path.getsize(output_save_path) / (1024 * 1024)
    print(f"✅ مدل با موفقیت در {output_save_path} ذخیره شد.")
    print(f"حجم فایل ذخیره شده: {file_size_mb:.2f} MB")
    
    # ----------------------------------------------------
    # و) چاپ معماری (اختیاری)
    # ----------------------------------------------------
    print("\n" + "="*70)
    print("معماری نهایی مدل هرس‌شده (ResNet_50_pruned_hardfakevsreal)")
    print("="*70)
    print(model_pruned)
    
except Exception as e:
    print(f"❌ خطا در لود یا ذخیره مدل: {e}")
