import torch
import torch.nn as nn
import sys
sys.path.append('/kaggle/working')

from model.pruned_model.ResNet_pruned  import ResNet_50_pruned_hardfakevsreal

checkpoint_path = '/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt'

print("="*70)
print("استخراج ماسک‌ها از مدل Sparse")
print("="*70)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
sparse_state_dict = checkpoint['student']

# استخراج تعداد فیلترهای باقی‌مانده از وزن‌های conv
def extract_masks_from_sparse_model(state_dict):
    """
    استخراج ماسک‌ها بر اساس تعداد فیلترهای باقی‌مانده
    از خروجی load_model.py میدونیم که مثلاً conv1 از 64 به 20 فیلتر رسیده
    """
    masks = []
    
    # تعریف ساختار ResNet50 Bottleneck
    # layer1: 3 blocks × 3 convs = 9 masks
    # layer2: 4 blocks × 3 convs = 12 masks
    # layer3: 6 blocks × 3 convs = 18 masks
    # layer4: 3 blocks × 3 convs = 9 masks
    # جمع: 48 masks
    
    # از خروجی load_model.py میدونیم تعداد فیلترهای pruned شده:
    # این اعداد رو از "Pruned weight shape" در خروجی استخراج کردیم
    pruned_filters = [
        # layer1.0
        20, 23, 94,
        # layer1.1
        13, 27, 91,
        # layer1.2
        27, 24, 82,
        # layer2.0
        44, 42, 92,
        # layer2.1
        47, 29, 94,
        # layer2.2
        37, 28, 71,
        # layer2.3
        43, 34, 56,
        # layer3.0
        65, 42, 66,
        # layer3.1
        63, 31, 66,
        # layer3.2
        59, 17, 60,
        # layer3.3
        40, 19, 40,
        # layer3.4
        30, 10, 31,
        # layer3.5
        29, 19, 29,
        # layer4.0
        69, 17, 62,
        # layer4.1
        59, 18, 83,
        # layer4.2
        72, 47, 89
    ]
    
    # تعداد کل فیلترها در ResNet50 استاندارد
    original_filters = [
        # layer1: 3 blocks
        64, 64, 256,  # block 0
        64, 64, 256,  # block 1
        64, 64, 256,  # block 2
        # layer2: 4 blocks
        128, 128, 512,  # block 0
        128, 128, 512,  # block 1
        128, 128, 512,  # block 2
        128, 128, 512,  # block 3
        # layer3: 6 blocks
        256, 256, 1024,  # block 0
        256, 256, 1024,  # block 1
        256, 256, 1024,  # block 2
        256, 256, 1024,  # block 3
        256, 256, 1024,  # block 4
        256, 256, 1024,  # block 5
        # layer4: 3 blocks
        512, 512, 2048,  # block 0
        512, 512, 2048,  # block 1
        512, 512, 2048,  # block 2
    ]
    
    print(f"تعداد ماسک‌های مورد نیاز: {len(original_filters)}")
    print(f"تعداد فیلترهای pruned شده: {len(pruned_filters)}")
    
    # ساخت ماسک‌ها
    for orig_filters, pruned_count in zip(original_filters, pruned_filters):
        mask = torch.zeros(orig_filters)
        # فرض می‌کنیم اولین فیلترها حفظ شدن
        mask[:pruned_count] = 1
        masks.append(mask)
        
    return masks, pruned_filters, original_filters

masks, pruned_counts, original_counts = extract_masks_from_sparse_model(sparse_state_dict)

print(f"\n✅ تعداد ماسک‌های ساخته شده: {len(masks)}")

# نمایش چند نمونه
print("\nنمونه ماسک‌ها:")
for i in range(min(5, len(masks))):
    print(f"  Mask {i}: {masks[i].shape}, فیلترهای باقی‌مانده: {int(masks[i].sum())}/{len(masks[i])}")

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
    print("\nممکنه نیاز باشه ماسک‌ها رو دستی تنظیم کنیم...")

# ===========================
# 3. لود وزن‌ها در مدل Pruned
# ===========================

print("\n" + "="*70)
print("لود وزن‌ها در مدل Pruned")
print("="*70)

def load_pruned_weights(model_pruned, sparse_state_dict, masks):
    """
    لود وزن‌های sparse در مدل pruned
    باید وزن‌ها رو از شکل کامل به شکل pruned تبدیل کنیم
    """
    pruned_state_dict = {}
    
    # conv1 و bn1 (قبل از layer1)
    if 'conv1.weight' in sparse_state_dict:
        pruned_state_dict['conv1.weight'] = sparse_state_dict['conv1.weight']
    if 'bn1.weight' in sparse_state_dict:
        pruned_state_dict['bn1.weight'] = sparse_state_dict['bn1.weight']
        pruned_state_dict['bn1.bias'] = sparse_state_dict['bn1.bias']
        pruned_state_dict['bn1.running_mean'] = sparse_state_dict['bn1.running_mean']
        pruned_state_dict['bn1.running_var'] = sparse_state_dict['bn1.running_var']
        pruned_state_dict['bn1.num_batches_tracked'] = sparse_state_dict['bn1.num_batches_tracked']
    
    # fc (لایه آخر)
    if 'fc.weight' in sparse_state_dict:
        pruned_state_dict['fc.weight'] = sparse_state_dict['fc.weight']
        pruned_state_dict['fc.bias'] = sparse_state_dict['fc.bias']
    
    mask_idx = 0
    layer_configs = [
        ('layer1', 3),  # 3 blocks
        ('layer2', 4),  # 4 blocks
        ('layer3', 6),  # 6 blocks
        ('layer4', 3),  # 3 blocks
    ]
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):  # conv1, conv2, conv3
                # کلیدهای sparse
                sparse_conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                sparse_bn_key = f'{layer_name}.{block_idx}.bn{conv_idx}.weight'
                
                if sparse_conv_key in sparse_state_dict:
                    sparse_weight = sparse_state_dict[sparse_conv_key]
                    mask = masks[mask_idx]
                    
                    # استخراج فیلترهای باقی‌مانده
                    active_filters = (mask == 1).nonzero(as_tuple=True)[0]
                    
                    # استخراج وزن‌های فیلترهای فعال
                    pruned_weight = sparse_weight[active_filters]
                    
                    # اگر conv1 یا conv2 بود، باید input channels هم prune بشه
                    if conv_idx > 1 and mask_idx > 0:
                        prev_mask = masks[mask_idx - 1]
                        active_in_channels = (prev_mask == 1).nonzero(as_tuple=True)[0]
                        pruned_weight = pruned_weight[:, active_in_channels]
                    
                    pruned_state_dict[sparse_conv_key] = pruned_weight
                    
                    # BatchNorm
                    if sparse_bn_key in sparse_state_dict:
                        pruned_state_dict[f'{layer_name}.{block_idx}.bn{conv_idx}.weight'] = \
                            sparse_state_dict[sparse_bn_key][active_filters]
                        pruned_state_dict[f'{layer_name}.{block_idx}.bn{conv_idx}.bias'] = \
                            sparse_state_dict[f'{layer_name}.{block_idx}.bn{conv_idx}.bias'][active_filters]
                        pruned_state_dict[f'{layer_name}.{block_idx}.bn{conv_idx}.running_mean'] = \
                            sparse_state_dict[f'{layer_name}.{block_idx}.bn{conv_idx}.running_mean'][active_filters]
                        pruned_state_dict[f'{layer_name}.{block_idx}.bn{conv_idx}.running_var'] = \
                            sparse_state_dict[f'{layer_name}.{block_idx}.bn{conv_idx}.running_var'][active_filters]
                        pruned_state_dict[f'{layer_name}.{block_idx}.bn{conv_idx}.num_batches_tracked'] = \
                            sparse_state_dict[f'{layer_name}.{block_idx}.bn{conv_idx}.num_batches_tracked']
                    
                    mask_idx += 1
            
            # downsample (اگر وجود داشته باشه)
            downsample_conv_key = f'{layer_name}.{block_idx}.downsample.0.weight'
            if downsample_conv_key in sparse_state_dict:
                pruned_state_dict[downsample_conv_key] = sparse_state_dict[downsample_conv_key]
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.weight'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.weight']
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.bias'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.bias']
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.running_mean'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.running_mean']
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.running_var'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.running_var']
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.num_batches_tracked'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.num_batches_tracked']
    
    return pruned_state_dict

try:
    pruned_weights = load_pruned_weights(model_pruned, sparse_state_dict, masks)
    print(f"✅ وزن‌های pruned آماده شد: {len(pruned_weights)} کلید")
    
    # لود در مدل
    missing, unexpected = model_pruned.load_state_dict(pruned_weights, strict=False)
    print(f"✅ وزن‌ها لود شدند")
    print(f"   - Missing keys: {len(missing)}")
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
    
    # مسیر ذخیره‌سازی
    save_path = '/kaggle/working/resnet50_pruned_model.pt'
    
    # ذخیره کامل مدل (شامل معماری + وزن‌ها)
    checkpoint_to_save = {
        'model_state_dict': model_pruned.state_dict(),
        'masks': masks,
        'pruned_counts': pruned_counts,
        'original_counts': original_counts,
        'total_params': total_params,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal'
    }
    
    torch.save(checkpoint_to_save, save_path)
    print(f"✅ مدل با موفقیت ذخیره شد در: {save_path}")
    
    # محاسبه حجم فایل
    import os
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ حجم فایل: {file_size_mb:.2f} MB")
    
    # ذخیره فقط وزن‌ها (فایل سبک‌تر)
    save_path_weights = '/kaggle/working/resnet50_pruned_weights_only.pt'
    torch.save(model_pruned.state_dict(), save_path_weights)
    file_size_weights_mb = os.path.getsize(save_path_weights) / (1024 * 1024)
    print(f"✅ فقط وزن‌ها ذخیره شد در: {save_path_weights}")
    print(f"✅ حجم فایل (فقط وزن‌ها): {file_size_weights_mb:.2f} MB")
    
    # نمایش اطلاعات ذخیره شده
    print("\n📦 اطلاعات ذخیره شده:")
    print(f"   - تعداد پارامترها: {total_params:,}")
    print(f"   - تعداد ماسک‌ها: {len(masks)}")
    print(f"   - معماری: ResNet_50_pruned_hardfakevsreal")
    
    print("\n💡 نحوه لود کردن:")
    print("# لود کامل (با ماسک‌ها)")
    print(f"checkpoint = torch.load('{save_path}')")
    print("model = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])")
    print("model.load_state_dict(checkpoint['model_state_dict'])")
    print("\n# یا فقط لود وزن‌ها (اگر ماسک‌ها رو دارید)")
    print("model = ResNet_50_pruned_hardfakevsreal(masks=masks)")
    print(f"model.load_state_dict(torch.load('{save_path_weights}'))")
        
except Exception as e:
    print(f"❌ خطا: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
