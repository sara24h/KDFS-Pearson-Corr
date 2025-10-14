import torch
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

# 1. مسیر و لود فایل چک‌پوینت
save_path = '/kaggle/input/10k_pruned_model_resnet50/pytorch/default/1/resnet50_pruned_model_learnable_masks.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    # لود چک‌پوینت کامل
    checkpoint_loaded = torch.load(save_path, map_location=device)

    # 2. استخراج اطلاعات کلیدی
    model_state_dict = checkpoint_loaded['model_state_dict']
    masks = checkpoint_loaded['masks']
    
    # 3. ساخت مدل هرس‌شده با استفاده از ماسک‌ها
    model_pruned = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    # 4. لود وزن‌های هرس‌شده
    model_pruned.load_state_dict(model_state_dict)
    
    model_pruned = model_pruned.to(device)
    model_pruned.eval()
    
    print("✅ مدل هرس‌شده با موفقیت بازسازی و لود شد!")
    print(f"تعداد پارامترها: {sum(p.numel() for p in model_pruned.parameters()):,}")

    print("\n" + "="*70)
    print("معماری نهایی مدل هرس‌شده (ResNet_50_pruned_hardfakevsreal)")
    print("="*70)

    print(model_pruned)
    
    print("\n" + "="*70)
    print("توجه: ابعاد هر لایه، معماری فشرده‌شده (هرس‌شده) را نشان می‌دهد.")
    print("="*70)

except Exception as e:
    print(f"❌ خطا در لود مدل: {e}")
    # اگر این خطا را گرفتید، مطمئن شوید که فایل ResNet_50_pruned_hardfakevsreal در مسیر sys.path قرار دارد.
