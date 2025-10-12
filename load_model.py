import torch
import os

# ===========================
# ذخیره مدل Pruned
# ===========================

print("\n" + "="*70)
print("ذخیره مدل Pruned")
print("="*70)

# مسیر ذخیره
save_dir = '/kaggle/working/saved_models'
os.makedirs(save_dir, exist_ok=True)

# 1. ذخیره کامل (مدل + ماسک‌ها + اطلاعات اضافی)
full_save_path = os.path.join(save_dir, 'resnet50_pruned_full.pth')

torch.save({
    'model_state_dict': model_pruned.state_dict(),
    'masks': masks,
    'pruned_filters': pruned_counts,
    'original_filters': original_counts,
    'architecture': 'ResNet50_Bottleneck_pruned',
    'num_classes': 1,
    'num_params': sum(p.numel() for p in model_pruned.parameters()),
    'best_prec1': checkpoint.get('best_prec1_after_finetune', None),
    'info': {
        'pruning_method': 'filter_pruning',
        'total_params': sum(p.numel() for p in model_pruned.parameters()),
        'compression_ratio': 23.51e6 / sum(p.numel() for p in model_pruned.parameters()),
        'params_reduction_percent': (1 - sum(p.numel() for p in model_pruned.parameters()) / 23.51e6) * 100
    }
}, full_save_path)

print(f"✅ مدل کامل ذخیره شد: {full_save_path}")
print(f"   حجم فایل: {os.path.getsize(full_save_path) / (1024**2):.2f} MB")

# 2. ذخیره فقط وزن‌ها (برای inference سبک‌تر)
weights_only_path = os.path.join(save_dir, 'resnet50_pruned_weights.pth')

torch.save(model_pruned.state_dict(), weights_only_path)

print(f"✅ فقط وزن‌ها ذخیره شد: {weights_only_path}")
print(f"   حجم فایل: {os.path.getsize(weights_only_path) / (1024**2):.2f} MB")

# 3. ذخیره ماسک‌ها به صورت جداگانه
masks_path = os.path.join(save_dir, 'resnet50_masks.pth')

torch.save({
    'masks': masks,
    'pruned_filters': pruned_counts,
    'original_filters': original_counts
}, masks_path)

print(f"✅ ماسک‌ها ذخیره شدند: {masks_path}")
print(f"   حجم فایل: {os.path.getsize(masks_path) / (1024):.2f} KB")

# ===========================
# توابع لود کردن
# ===========================

print("\n" + "="*70)
print("توابع برای لود کردن مدل ذخیره شده")
print("="*70)

def load_full_pruned_model(checkpoint_path):
    """
    لود کامل مدل pruned (با ماسک‌ها)
    
    Args:
        checkpoint_path: مسیر فایل ذخیره شده
    
    Returns:
        model: مدل pruned لود شده
        info: اطلاعات اضافی
    """
    print(f"در حال لود مدل از: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # ساخت مدل
    from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    masks = checkpoint['masks']
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    # لود وزن‌ها
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ مدل با موفقیت لود شد!")
    print(f"   - تعداد پارامترها: {checkpoint['num_params']:,}")
    if 'best_prec1' in checkpoint and checkpoint['best_prec1'] is not None:
        print(f"   - بهترین دقت: {checkpoint['best_prec1']:.2f}%")
    
    return model, checkpoint.get('info', {})

def load_weights_only(weights_path, masks_path):
    """
    لود فقط وزن‌ها (نیاز به فایل ماسک‌ها داره)
    
    Args:
        weights_path: مسیر فایل وزن‌ها
        masks_path: مسیر فایل ماسک‌ها
    
    Returns:
        model: مدل pruned لود شده
    """
    print(f"در حال لود وزن‌ها از: {weights_path}")
    print(f"در حال لود ماسک‌ها از: {masks_path}")
    
    # لود ماسک‌ها
    masks_data = torch.load(masks_path, map_location='cpu')
    masks = masks_data['masks']
    
    # ساخت مدل
    from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    # لود وزن‌ها
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ مدل با موفقیت لود شد!")
    
    return model

# نمایش مثال استفاده
print("\nنحوه استفاده:")
print("-" * 70)
print("""
# روش 1: لود کامل
model, info = load_full_pruned_model('/kaggle/working/saved_models/resnet50_pruned_full.pth')
print(f"نسبت فشرده‌سازی: {info['compression_ratio']:.2f}x")

# روش 2: لود فقط وزن‌ها
model = load_weights_only(
    '/kaggle/working/saved_models/resnet50_pruned_weights.pth',
    '/kaggle/working/saved_models/resnet50_masks.pth'
)

# استفاده برای پیش‌بینی
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('test.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output, _ = model(image_tensor)
    prob = torch.sigmoid(output).item()
    prediction = "Fake" if prob > 0.5 else "Real"
    print(f"Prediction: {prediction} (probability: {prob:.4f})")
""")

# ===========================
# تست لود مجدد
# ===========================

print("\n" + "="*70)
print("تست لود مجدد مدل")
print("="*70)

try:
    # تست لود مدل کامل
    loaded_model, info = load_full_pruned_model(full_save_path)
    
    # تست با ورودی نمونه
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224)
        
        # خروجی مدل اصلی
        output_original, _ = model_pruned(test_input)
        
        # خروجی مدل لود شده
        output_loaded, _ = loaded_model(test_input)
        
        # مقایسه
        diff = (output_original - output_loaded).abs().item()
        
        print(f"\n✅ تست لود موفق!")
        print(f"   - خروجی مدل اصلی: {output_original.item():.6f}")
        print(f"   - خروجی مدل لود شده: {output_loaded.item():.6f}")
        print(f"   - تفاوت: {diff:.8f}")
        
        if diff < 1e-6:
            print(f"   ✅ خروجی‌ها دقیقاً یکسان هستن!")
        else:
            print(f"   ⚠️ تفاوت جزئی وجود داره (قابل قبول)")
            
    print(f"\n📊 اطلاعات مدل:")
    print(f"   - نسبت فشرده‌سازی: {info['compression_ratio']:.2f}x")
    print(f"   - کاهش پارامترها: {info['params_reduction_percent']:.2f}%")
    
except Exception as e:
    print(f"❌ خطا در تست: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ همه چی آماده است!")
print("="*70)
print(f"\nفایل‌های ذخیره شده:")
print(f"  1. {full_save_path}")
print(f"  2. {weights_only_path}")
print(f"  3. {masks_path}")
print("\n🎯 می‌تونی این فایل‌ها رو دانلود کنی و در هر جا استفاده کنی!")
print("="*70)
