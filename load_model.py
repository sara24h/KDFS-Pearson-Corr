import torch
import torch.nn as nn
import sys
sys.path.append('/kaggle/working')

from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

checkpoint_path = '/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt'

# ===========================
# 1. استخراج ماسک‌ها از مدل Sparse
# ===========================

print("="*70)
print("مرحله 1: استخراج ماسک‌ها از مدل Sparse")
print("="*70)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
sparse_state_dict = checkpoint['student']

def extract_masks_from_model(model):
    """
    استخراج ماسک‌های رسمی از mask_modules مدل sparse
    """
    masks = []
    remaining_counts = []
    
    print("\nاستخراج ماسک‌ها از mask_modules:\n")
    
    for i, mask_module in enumerate(model.mask_modules):
        # فرض: mask_weight شکلی شبیه [1, C, 1, 1] دارد
        mask_weight = mask_module.mask_weight  # این یک پارامتر یادگیری‌شده است
        # تبدیل به ماسک باینری: معمولاً با argmax یا threshold
        # در بسیاری از روش‌ها، mask_weight دو کانال دارد: [keep, prune]
        if mask_weight.shape[1] == 2:  # binary mask (keep/prune)
            mask_binary = torch.argmax(mask_weight, dim=1).squeeze()  # 0 یا 1
        else:
            # یا اگر مستقیماً ماسک باشد (مثلاً با sigmoid)، آستانه بگذارید
            mask_binary = (mask_weight.squeeze() > 0.5).float()
        
        # تبدیل به فرمت مورد نیاز (float یا bool)
        mask = mask_binary.float()
        remaining = int(mask.sum().item())
        original = mask.shape[0]
        
        masks.append(mask)
        remaining_counts.append(remaining)
        
        print(f"Mask {i}: {remaining}/{original} فیلتر فعال")
    
    return masks, remaining_counts

masks, remaining_counts = extract_masks_from_model(sparse_state_dict)

print(f"\n✅ تعداد ماسک‌های ساخته شده: {len(masks)}")
print(f"✅ جمع فیلترهای باقی‌مانده: {sum(remaining_counts)}")

# نمایش لیست
print("\n" + "="*70)
print("📋 فیلترهای باقی‌مانده:")
print("="*70)
print("remaining_filters = [")

layer_names = []
for i in range(1, 5):
    num_blocks = [3, 4, 6, 3][i-1]
    for j in range(num_blocks):
        layer_names.append(f'layer{i}.{j}')

idx = 0
for layer_name in layer_names:
    layer_data = remaining_counts[idx:idx+3]
    print(f"    # {layer_name}")
    print(f"    {', '.join(map(str, layer_data))},")
    idx += 3
print("]")

# ===========================
# 2. ساخت مدل Pruned
# ===========================

print("\n" + "="*70)
print("مرحله 2: ساخت مدل Pruned")
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
    raise

# ===========================
# 3. لود وزن‌ها در مدل Pruned
# ===========================

print("\n" + "="*70)
print("مرحله 3: لود وزن‌ها در مدل Pruned")
print("="*70)

def load_pruned_weights(model_pruned, sparse_state_dict, masks):
    """
    لود وزن‌های sparse در مدل pruned
    فقط فیلترهای active رو استخراج و لود می‌کنیم
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
    
    print("\nلود وزن‌ها:\n")
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):  # conv1, conv2, conv3
                # کلیدهای sparse
                sparse_conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                sparse_bn_key = f'{layer_name}.{block_idx}.bn{conv_idx}.weight'
                
                if sparse_conv_key in sparse_state_dict:
                    sparse_weight = sparse_state_dict[sparse_conv_key]
                    mask = masks[mask_idx]
                    
                    # استخراج فیلترهای فعال (active)
                    active_filters = (mask == 1).nonzero(as_tuple=True)[0]
                    
                    # استخراج وزن‌های فیلترهای فعال
                    pruned_weight = sparse_weight[active_filters]
                    
                    # اگر conv2 یا conv3 بود، باید input channels هم بر اساس ماسک قبلی prune بشه
                    if conv_idx > 1 and mask_idx > 0:
                        prev_mask = masks[mask_idx - 1]
                        active_in_channels = (prev_mask == 1).nonzero(as_tuple=True)[0]
                        pruned_weight = pruned_weight[:, active_in_channels]
                    
                    pruned_state_dict[sparse_conv_key] = pruned_weight
                    
                    print(f"{layer_name}.{block_idx}.conv{conv_idx}: {pruned_weight.shape}")
                    
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
                # downsample معمولاً prune نمیشه، پس مستقیم کپی می‌کنیم
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
    print(f"\n✅ وزن‌های pruned آماده شد: {len(pruned_weights)} کلید")
    
    # لود در مدل
    missing, unexpected = model_pruned.load_state_dict(pruned_weights, strict=False)
    print(f"✅ وزن‌ها لود شدند")
    print(f"   - Missing keys: {len(missing)}")
    print(f"   - Unexpected keys: {len(unexpected)}")
    
    if len(missing) > 0:
        print(f"   - نمونه missing keys: {missing[:3]}")
    if len(unexpected) > 0:
        print(f"   - نمونه unexpected keys: {unexpected[:3]}")
    
except Exception as e:
    print(f"❌ خطا در لود وزن‌ها: {e}")
    import traceback
    traceback.print_exc()
    raise

# ===========================
# 4. تست مدل
# ===========================

print("\n" + "="*70)
print("مرحله 4: تست مدل Pruned")
print("="*70)

try:
    model_pruned.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224)
        output, features = model_pruned(dummy_input)
        print(f"✅ تست موفق!")
        print(f"   - شکل input: {dummy_input.shape}")
        print(f"   - شکل output: {output.shape}")
        print(f"   - تعداد feature maps: {len(features)}")
        if len(features) > 0:
            print(f"   - شکل اولین feature map: {features[0].shape}")
except Exception as e:
    print(f"❌ خطا در تست: {e}")
    import traceback
    traceback.print_exc()
    raise

# ===========================
# 5. ذخیره مدل Pruned
# ===========================

print("\n" + "="*70)
print("مرحله 5: ذخیره مدل Pruned")
print("="*70)

try:
    # مسیر ذخیره‌سازی
    save_path = '/kaggle/working/resnet50_pruned_model.pt'
    
    # ذخیره کامل مدل (شامل معماری + وزن‌ها + ماسک‌ها)
    checkpoint_to_save = {
        'model_state_dict': model_pruned.state_dict(),
        'masks': masks,
        'remaining_counts': remaining_counts,
        'total_params': total_params,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal',
        'best_prec1': checkpoint.get('best_prec1_after_finetune', None)
    }
    
    torch.save(checkpoint_to_save, save_path)
    print(f"✅ مدل کامل ذخیره شد در: {save_path}")
    
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
    print(f"   - جمع فیلترهای باقی‌مانده: {sum(remaining_counts)}")
    print(f"   - معماری: ResNet_50_pruned_hardfakevsreal")
    if checkpoint_to_save['best_prec1']:
        print(f"   - بهترین دقت: {checkpoint_to_save['best_prec1']:.2f}%")
    
    print("\n💡 نحوه لود کردن:")
    print("# روش 1: لود کامل (با ماسک‌ها)")
    print(f"checkpoint = torch.load('{save_path}')")
    print("masks = checkpoint['masks']")
    print("model = ResNet_50_pruned_hardfakevsreal(masks=masks)")
    print("model.load_state_dict(checkpoint['model_state_dict'])")
    
    print("\n# روش 2: فقط لود وزن‌ها (اگر ماسک‌ها رو دارید)")
    print("model = ResNet_50_pruned_hardfakevsreal(masks=masks)")
    print(f"model.load_state_dict(torch.load('{save_path_weights}'))")
    
except Exception as e:
    print(f"❌ خطا در ذخیره: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("🎉 تمام مراحل با موفقیت انجام شد!")
print("="*70)
