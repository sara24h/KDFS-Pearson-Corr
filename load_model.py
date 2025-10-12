import torch
import torch.nn as nn
import sys
sys.path.append('/kaggle/working')

from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

checkpoint_path = '/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt'

print("="*70)
print("استخراج خودکار ماسک‌ها از مدل Sparse")
print("="*70)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
sparse_state_dict = checkpoint['student']

def extract_masks_automatically(state_dict):
    """
    استخراج خودکار ماسک‌ها بر اساس shape واقعی وزن‌های conv در checkpoint
    """
    masks = []
    pruned_filters = []
    original_filters = []
    
    # تعریف ساختار ResNet50 - تعداد فیلترهای استاندارد
    resnet50_structure = {
        'layer1': {'blocks': 3, 'filters': [64, 64, 256]},
        'layer2': {'blocks': 4, 'filters': [128, 128, 512]},
        'layer3': {'blocks': 6, 'filters': [256, 256, 1024]},
        'layer4': {'blocks': 3, 'filters': [512, 512, 2048]}
    }
    
    print("\n🔍 در حال تحلیل checkpoint...")
    
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        num_blocks = resnet50_structure[layer_name]['blocks']
        standard_filters = resnet50_structure[layer_name]['filters']
        
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):  # conv1, conv2, conv3
                conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                
                if conv_key in state_dict:
                    # گرفتن shape واقعی وزن
                    weight_shape = state_dict[conv_key].shape
                    pruned_count = weight_shape[0]  # تعداد فیلترهای خروجی (out_channels)
                    original_count = standard_filters[conv_idx - 1]
                    
                    # ساخت ماسک
                    mask = torch.zeros(original_count)
                    mask[:pruned_count] = 1
                    
                    masks.append(mask)
                    pruned_filters.append(pruned_count)
                    original_filters.append(original_count)
                    
                    print(f"  ✓ {conv_key}: {pruned_count}/{original_count} فیلتر باقی‌مانده")
                else:
                    print(f"  ⚠ {conv_key} در checkpoint یافت نشد!")
    
    return masks, pruned_filters, original_filters

# استخراج خودکار
masks, pruned_counts, original_counts = extract_masks_automatically(sparse_state_dict)

print(f"\n✅ تعداد ماسک‌های ساخته شده: {len(masks)}")
print(f"📊 آمار کلی:")
print(f"   - مجموع فیلترهای اصلی: {sum(original_counts):,}")
print(f"   - مجموع فیلترهای باقی‌مانده: {sum(pruned_counts):,}")
print(f"   - نرخ حذف: {(1 - sum(pruned_counts)/sum(original_counts))*100:.2f}%")

# نمایش چند نمونه
print("\n📋 نمونه ماسک‌ها:")
for i in range(min(10, len(masks))):
    remaining = int(masks[i].sum())
    total = len(masks[i])
    print(f"  Mask {i:2d}: {remaining:4d}/{total:4d} ({remaining/total*100:5.1f}%)")

# ===========================
# 2. ساخت مدل Pruned
# ===========================

print("\n" + "="*70)
print("ساخت مدل Pruned")
print("="*70)

try:
    model_pruned = ResNet_50_pruned_hardfakevsreal(masks=masks)
    print("✅ مدل pruned با موفقیت ساخته شد!")
    
    # محاسبه تعداد پارامترها
    total_params = sum(p.numel() for p in model_pruned.parameters())
    print(f"✅ تعداد پارامترهای مدل pruned: {total_params:,}")
    
except Exception as e:
    print(f"❌ خطا در ساخت مدل: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===========================
# 3. لود وزن‌ها در مدل Pruned
# ===========================

print("\n" + "="*70)
print("لود وزن‌ها در مدل Pruned")
print("="*70)

def load_pruned_weights(model_pruned, sparse_state_dict, masks):
    """
    لود وزن‌های sparse در مدل pruned
    """
    pruned_state_dict = {}
    
    # conv1 و bn1 (قبل از layer1)
    for key in ['conv1.weight', 'bn1.weight', 'bn1.bias', 
                'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked']:
        if key in sparse_state_dict:
            pruned_state_dict[key] = sparse_state_dict[key]
    
    # fc (لایه آخر)
    for key in ['fc.weight', 'fc.bias']:
        if key in sparse_state_dict:
            pruned_state_dict[key] = sparse_state_dict[key]
    
    mask_idx = 0
    layer_configs = [
        ('layer1', 3), ('layer2', 4), ('layer3', 6), ('layer4', 3)
    ]
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):  # conv1, conv2, conv3
                sparse_conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                
                if sparse_conv_key in sparse_state_dict:
                    sparse_weight = sparse_state_dict[sparse_conv_key]
                    mask = masks[mask_idx]
                    
                    # استخراج فیلترهای باقی‌مانده
                    active_filters = (mask == 1).nonzero(as_tuple=True)[0]
                    pruned_weight = sparse_weight[active_filters]
                    
                    # اگر conv2 یا conv3، input channels هم باید prune بشه
                    if conv_idx > 1 and mask_idx > 0:
                        prev_mask = masks[mask_idx - 1]
                        active_in_channels = (prev_mask == 1).nonzero(as_tuple=True)[0]
                        pruned_weight = pruned_weight[:, active_in_channels]
                    
                    pruned_state_dict[sparse_conv_key] = pruned_weight
                    
                    # BatchNorm
                    bn_prefix = f'{layer_name}.{block_idx}.bn{conv_idx}'
                    for bn_key in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                        full_key = f'{bn_prefix}.{bn_key}'
                        if full_key in sparse_state_dict:
                            if bn_key == 'num_batches_tracked':
                                pruned_state_dict[full_key] = sparse_state_dict[full_key]
                            else:
                                pruned_state_dict[full_key] = sparse_state_dict[full_key][active_filters]
                    
                    mask_idx += 1
            
            # downsample (اگر وجود داشته باشه)
            downsample_prefix = f'{layer_name}.{block_idx}.downsample'
            if f'{downsample_prefix}.0.weight' in sparse_state_dict:
                for key in sparse_state_dict.keys():
                    if key.startswith(downsample_prefix):
                        pruned_state_dict[key] = sparse_state_dict[key]
    
    return pruned_state_dict

try:
    pruned_weights = load_pruned_weights(model_pruned, sparse_state_dict, masks)
    print(f"✅ وزن‌های pruned آماده شد: {len(pruned_weights)} کلید")
    
    # لود در مدل
    missing, unexpected = model_pruned.load_state_dict(pruned_weights, strict=False)
    print(f"✅ وزن‌ها لود شدند")
    print(f"   - Missing keys: {len(missing)}")
    if len(missing) > 0:
        print(f"   - اولین missing keys: {missing[:5]}")
    print(f"   - Unexpected keys: {len(unexpected)}")
    
    # تست
    model_pruned.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224)
        output, features = model_pruned(dummy_input)
        print(f"\n✅ تست موفق!")
        print(f"   - شکل خروجی: {output.shape}")
        print(f"   - تعداد feature maps: {len(features)}")
    
    # ===========================
    # 4. ذخیره مدل Pruned
    # ===========================
    
    print("\n" + "="*70)
    print("ذخیره مدل Pruned")
    print("="*70)
    
    save_path = '/kaggle/working/resnet50_pruned_model.pt'
    
    checkpoint_to_save = {
        'model_state_dict': model_pruned.state_dict(),
        'masks': masks,
        'pruned_counts': pruned_counts,
        'original_counts': original_counts,
        'total_params': total_params,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal',
        'compression_ratio': sum(pruned_counts) / sum(original_counts)
    }
    
    torch.save(checkpoint_to_save, save_path)
    print(f"✅ مدل با موفقیت ذخیره شد در: {save_path}")
    
    # محاسبه حجم فایل
    import os
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ حجم فایل: {file_size_mb:.2f} MB")
    
    # ذخیره فقط وزن‌ها
    save_path_weights = '/kaggle/working/resnet50_pruned_weights_only.pt'
    torch.save(model_pruned.state_dict(), save_path_weights)
    file_size_weights_mb = os.path.getsize(save_path_weights) / (1024 * 1024)
    print(f"✅ فقط وزن‌ها ذخیره شد در: {save_path_weights}")
    print(f"✅ حجم فایل (فقط وزن‌ها): {file_size_weights_mb:.2f} MB")
    
    print("\n📦 اطلاعات ذخیره شده:")
    print(f"   - تعداد پارامترها: {total_params:,}")
    print(f"   - تعداد ماسک‌ها: {len(masks)}")
    print(f"   - نسبت فشرده‌سازی: {checkpoint_to_save['compression_ratio']:.2%}")
    print(f"   - معماری: ResNet_50_pruned_hardfakevsreal")
    
    print("\n💡 نحوه لود کردن:")
    print("```python")
    print("checkpoint = torch.load('resnet50_pruned_model.pt')")
    print("model = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])")
    print("model.load_state_dict(checkpoint['model_state_dict'])")
    print("model.eval()")
    print("```")
        
except Exception as e:
    print(f"❌ خطا: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ فرآیند با موفقیت کامل شد!")
print("="*70)
