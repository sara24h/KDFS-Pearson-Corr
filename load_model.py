import torch
import numpy as np
import matplotlib.pyplot as plt

# بارگذاری مدل
model_path = '/kaggle/input/fuzzy-ranked-based-ensemble/resnet50_pruned_model.pt'
checkpoint = torch.load(model_path, map_location='cpu')

print("="*100)
print("تحلیل جامع مدل ResNet50 Pruned")
print("="*100)

# استخراج اطلاعات اصلی
state_dict = checkpoint['model_state_dict']
masks = checkpoint['masks']
pruned_counts = checkpoint['pruned_counts']
original_counts = checkpoint['original_counts']
total_params = checkpoint['total_params']
model_arch = checkpoint['model_architecture']

print(f"\n📊 معماری مدل: {model_arch}")
print(f"📊 تعداد کل پارامترها (metadata): {total_params:,}")

print("\n" + "="*100)
print("تحلیل ماسک‌های Pruning")
print("="*100)

if masks and len(masks) > 0:
    print(f"\n✓ تعداد ماسک‌ها: {len(masks)}")
    print(f"✓ تعداد لایه‌های pruned شده: {len(pruned_counts)}")
    print(f"✓ تعداد لایه‌های اصلی: {len(original_counts)}")
    
    # محاسبه آمار کلی
    total_pruned = sum(pruned_counts)
    total_original = sum(original_counts)
    overall_pruning_ratio = (total_pruned / total_original) * 100 if total_original > 0 else 0
    
    print(f"\n📉 آمار Pruning کلی:")
    print(f"   - پارامترهای اصلی: {total_original:,}")
    print(f"   - پارامترهای حذف شده: {total_pruned:,}")
    print(f"   - پارامترهای باقیمانده: {total_original - total_pruned:,}")
    print(f"   - نرخ Pruning: {overall_pruning_ratio:.2f}%")
    print(f"   - نرخ فشرده‌سازی: {(total_original / (total_original - total_pruned)):.2f}x")
    
    # تحلیل جزئیات هر ماسک
    print(f"\n{'#':<5} {'Layer Name':<50} {'اصلی':<15} {'حذف شده':<15} {'باقیمانده':<15} {'نرخ Pruning':<15}")
    print("-"*115)
    
    layer_names = [name for name in state_dict.keys() if 'weight' in name and 'bn' not in name and 'downsample' not in name]
    
    for idx, (mask, original, pruned) in enumerate(zip(masks, original_counts, pruned_counts)):
        remaining = original - pruned
        pruning_ratio = (pruned / original) * 100 if original > 0 else 0
        
        # پیدا کردن نام لایه مرتبط
        layer_name = f"Layer {idx+1}"
        if idx < len(layer_names):
            layer_name = layer_names[idx]
        
        print(f"{idx+1:<5} {layer_name:<50} {original:<15,} {pruned:<15,} {remaining:<15,} {pruning_ratio:<15.2f}%")
    
    # لایه‌های با بیشترین pruning
    print("\n" + "="*100)
    print("🎯 لایه‌های با بیشترین Pruning")
    print("="*100)
    
    pruning_ratios = [(i, (p/o)*100 if o > 0 else 0) for i, (p, o) in enumerate(zip(pruned_counts, original_counts))]
    top_pruned = sorted(pruning_ratios, key=lambda x: x[1], reverse=True)[:10]
    
    for rank, (idx, ratio) in enumerate(top_pruned, 1):
        layer_name = f"Layer {idx+1}"
        if idx < len(layer_names):
            layer_name = layer_names[idx]
        print(f"{rank}. {layer_name}: {ratio:.2f}% (حذف شده: {pruned_counts[idx]:,}/{original_counts[idx]:,})")
    
    # لایه‌های با کمترین pruning
    print("\n" + "="*100)
    print("🎯 لایه‌های با کمترین Pruning")
    print("="*100)
    
    bottom_pruned = sorted(pruning_ratios, key=lambda x: x[1])[:10]
    
    for rank, (idx, ratio) in enumerate(bottom_pruned, 1):
        layer_name = f"Layer {idx+1}"
        if idx < len(layer_names):
            layer_name = layer_names[idx]
        print(f"{rank}. {layer_name}: {ratio:.2f}% (حذف شده: {pruned_counts[idx]:,}/{original_counts[idx]:,})")
    
    # آمار توزیع pruning
    print("\n" + "="*100)
    print("📈 آمار توزیع Pruning")
    print("="*100)
    
    pruning_percentages = [(p/o)*100 if o > 0 else 0 for p, o in zip(pruned_counts, original_counts)]
    
    print(f"\nمیانگین نرخ pruning: {np.mean(pruning_percentages):.2f}%")
    print(f"میانه نرخ pruning: {np.median(pruning_percentages):.2f}%")
    print(f"انحراف معیار: {np.std(pruning_percentages):.2f}%")
    print(f"حداقل نرخ pruning: {np.min(pruning_percentages):.2f}%")
    print(f"حداکثر نرخ pruning: {np.max(pruning_percentages):.2f}%")
    
    # دسته‌بندی لایه‌ها بر اساس نرخ pruning
    ranges = [
        (0, 10, "کم (0-10%)"),
        (10, 30, "متوسط (10-30%)"),
        (30, 50, "زیاد (30-50%)"),
        (50, 70, "خیلی زیاد (50-70%)"),
        (70, 100, "شدید (70-100%)")
    ]
    
    print(f"\n📊 توزیع لایه‌ها بر اساس نرخ pruning:")
    for min_val, max_val, label in ranges:
        count = sum(1 for p in pruning_percentages if min_val <= p < max_val)
        percentage = (count / len(pruning_percentages)) * 100
        print(f"   {label}: {count} لایه ({percentage:.1f}%)")
    
    # تحلیل ماسک‌ها
    print("\n" + "="*100)
    print("🔍 تحلیل ساختار ماسک‌ها")
    print("="*100)
    
    print(f"\nنمونه‌ای از ماسک‌های اول:")
    for i in range(min(5, len(masks))):
        mask = masks[i]
        if isinstance(mask, (list, np.ndarray, torch.Tensor)):
            if isinstance(mask, torch.Tensor):
                mask_array = mask.cpu().numpy()
            else:
                mask_array = np.array(mask)
            
            unique_values = np.unique(mask_array)
            print(f"\nماسک {i+1}:")
            print(f"   - شکل: {mask_array.shape if hasattr(mask_array, 'shape') else len(mask_array)}")
            print(f"   - مقادیر منحصر به فرد: {unique_values}")
            print(f"   - تعداد 0ها: {np.sum(mask_array == 0)}")
            print(f"   - تعداد 1ها: {np.sum(mask_array == 1)}")

else:
    print("\n⚠ هیچ ماسک صریحی یافت نشد")

print("\n" + "="*100)
print("تحلیل کامل شد")
print("="*100)
