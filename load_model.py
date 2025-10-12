import torch
import torch.nn as nn
import sys
import os
sys.path.append('/kaggle/working')

# مدل‌های مورد نیاز
try:
    from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
    HAS_SPARSE_MODEL = True
except ImportError as e:
    print(f"⚠️ توجه: مدل sparse قابل import نیست: {e}")
    HAS_SPARSE_MODEL = False

from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

checkpoint_path = '/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt'

print("="*70)
print("مرحله 1: استخراج ماسک‌ها از مدل Sparse")
print("="*70)

# بارگذاری چک‌پوینت
checkpoint = torch.load(checkpoint_path, map_location='cpu')
sparse_state_dict = checkpoint['student']

masks = None
remaining_counts = None

# ==============================================================
# روش 1: استفاده از mask_modules (اگر مدل sparse در دسترس باشد)
# ==============================================================

if HAS_SPARSE_MODEL:
    try:
        print("\nتلاش برای استخراج ماسک‌ها از mask_modules...")
        sparse_model = ResNet_50_sparse_hardfakevsreal()
        sparse_model.load_state_dict(sparse_state_dict)
        
        if hasattr(sparse_model, 'mask_modules') and len(sparse_model.mask_modules) > 0:
            print(f"\n✅ مدل شامل {len(sparse_model.mask_modules)} mask_module است.")
            masks = []
            remaining_counts = []
            
            for i, mask_module in enumerate(sparse_model.mask_modules):
                mask_weight = mask_module.mask_weight  # یک پارامتر torch
                shape = mask_weight.shape
                
                # تشخیص نوع ماسک
                if len(shape) == 5 and shape[1] == 2:
                    # فرض: [1, 2, C, 1, 1] → binary selection
                    mask_binary = torch.argmax(mask_weight, dim=1).squeeze()  # [C]
                elif len(shape) == 4:
                    # فرض: [1, C, 1, 1] → continuous mask
                    mask_binary = (mask_weight.squeeze() > 0.5).float()
                else:
                    raise ValueError(f"شکل ناشناخته mask_weight: {shape}")
                
                mask = mask_binary.float()
                remaining = int(mask.sum().item())
                original = mask.shape[0]
                masks.append(mask)
                remaining_counts.append(remaining)
                print(f"Mask {i}: {remaining}/{original} فیلتر فعال")
            
            print(f"\n✅ ماسک‌ها از mask_modules با موفقیت استخراج شدند.")
        else:
            print("❌ مدل فاقد mask_modules است. استفاده از روش جایگزین (norm-based).")
            HAS_SPARSE_MODEL = False
    except Exception as e:
        print(f"❌ خطا در استفاده از mask_modules: {e}")
        HAS_SPARSE_MODEL = False

# ==============================================================
# روش 2: fallback به روش norm-based (اگر mask_modules وجود نداشت)
# ==============================================================

if masks is None:
    print("\nاستفاده از روش جایگزین: استخراج ماسک از نُرم فیلترها (norm-based)...")
    
    def extract_masks_from_weights(state_dict):
        masks = []
        remaining_counts = []
        layer_configs = [('layer1', 3), ('layer2', 4), ('layer3', 6), ('layer4', 3)]
        print("\nاستخراج ماسک‌ها:\n")
        for layer_name, num_blocks in layer_configs:
            for block_idx in range(num_blocks):
                for conv_idx in range(1, 4):
                    weight_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                    if weight_key in state_dict:
                        weight = state_dict[weight_key]
                        num_filters = weight.shape[0]
                        filter_norms = torch.norm(weight.view(num_filters, -1), p=1, dim=1)
                        mask = (filter_norms >= 1e-6).float()
                        remaining = int(mask.sum().item())
                        masks.append(mask)
                        remaining_counts.append(remaining)
                        print(f"{layer_name}.{block_idx}.conv{conv_idx}: {remaining}/{num_filters} فیلتر")
        return masks, remaining_counts

    masks, remaining_counts = extract_masks_from_weights(sparse_state_dict)

print(f"\n✅ تعداد ماسک‌های ساخته شده: {len(masks)}")
print(f"✅ جمع فیلترهای باقی‌مانده: {sum(remaining_counts)}")

# نمایش لیست فیلترهای باقی‌مانده
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
    total_params = sum(p.numel() for p in model_pruned.parameters())
    print("✅ مدل pruned با موفقیت ساخته شد!")
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
    pruned_state_dict = {}
    
    # conv1, bn1
    for key in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 
                'bn1.running_var', 'bn1.num_batches_tracked']:
        if key in sparse_state_dict:
            pruned_state_dict[key] = sparse_state_dict[key]
    
    # fc
    for key in ['fc.weight', 'fc.bias']:
        if key in sparse_state_dict:
            pruned_state_dict[key] = sparse_state_dict[key]
    
    mask_idx = 0
    layer_configs = [('layer1', 3), ('layer2', 4), ('layer3', 6), ('layer4', 3)]
    print("\nلود وزن‌ها:\n")
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):
                sparse_conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                sparse_bn_key = f'{layer_name}.{block_idx}.bn{conv_idx}.weight'
                
                if sparse_conv_key in sparse_state_dict:
                    sparse_weight = sparse_state_dict[sparse_conv_key]
                    mask = masks[mask_idx]
                    active_out = (mask == 1).nonzero(as_tuple=True)[0]
                    pruned_weight = sparse_weight[active_out]
                    
                    if conv_idx > 1 and mask_idx > 0:
                        prev_mask = masks[mask_idx - 1]
                        active_in = (prev_mask == 1).nonzero(as_tuple=True)[0]
                        pruned_weight = pruned_weight[:, active_in]
                    
                    pruned_state_dict[sparse_conv_key] = pruned_weight
                    print(f"{layer_name}.{block_idx}.conv{conv_idx}: {pruned_weight.shape}")
                    
                    # BatchNorm
                    if sparse_bn_key in sparse_state_dict:
                        for suffix in ['', '.bias', '.running_mean', '.running_var', '.num_batches_tracked']:
                            key = f'{layer_name}.{block_idx}.bn{conv_idx}' + suffix
                            if suffix == '':
                                key += '.weight'
                            if key in sparse_state_dict:
                                if 'num_batches_tracked' in key:
                                    pruned_state_dict[key] = sparse_state_dict[key]
                                else:
                                    pruned_state_dict[key] = sparse_state_dict[key][active_out]
                    mask_idx += 1
            
            # downsample
            downsample_conv_key = f'{layer_name}.{block_idx}.downsample.0.weight'
            if downsample_conv_key in sparse_state_dict:
                for suffix in ['.0.weight', '.1.weight', '.1.bias', '.1.running_mean', '.1.running_var', '.1.num_batches_tracked']:
                    key = f'{layer_name}.{block_idx}.downsample' + suffix
                    if key in sparse_state_dict:
                        pruned_state_dict[key] = sparse_state_dict[key]
    
    return pruned_state_dict

try:
    pruned_weights = load_pruned_weights(model_pruned, sparse_state_dict, masks)
    print(f"\n✅ وزن‌های pruned آماده شد: {len(pruned_weights)} کلید")
    missing, unexpected = model_pruned.load_state_dict(pruned_weights, strict=False)
    print(f"✅ وزن‌ها لود شدند")
    print(f"   - Missing keys: {len(missing)}")
    print(f"   - Unexpected keys: {len(unexpected)}")
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
        print(f"   - شکل output: {output.shape}")
        print(f"   - تعداد feature maps: {len(features)}")
        if features:
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
    save_path = '/kaggle/working/resnet50_pruned_model.pt'
    checkpoint_to_save = {
        'model_state_dict': model_pruned.state_dict(),
        'masks': masks,
        'remaining_counts': remaining_counts,
        'total_params': total_params,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal',
        'best_prec1': checkpoint.get('best_prec1_after_finetune', None)
    }
    torch.save(checkpoint_to_save, save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ مدل کامل ذخیره شد در: {save_path} ({file_size_mb:.2f} MB)")
    
    save_path_weights = '/kaggle/working/resnet50_pruned_weights_only.pt'
    torch.save(model_pruned.state_dict(), save_path_weights)
    file_size_weights_mb = os.path.getsize(save_path_weights) / (1024 * 1024)
    print(f"✅ فقط وزن‌ها ذخیره شد ({file_size_weights_mb:.2f} MB)")

    print("\n📦 اطلاعات ذخیره شده:")
    print(f"   - تعداد پارامترها: {total_params:,}")
    print(f"   - تعداد ماسک‌ها: {len(masks)}")
    print(f"   - جمع فیلترهای باقی‌مانده: {sum(remaining_counts)}")
    if checkpoint_to_save['best_prec1']:
        print(f"   - بهترین دقت: {checkpoint_to_save['best_prec1']:.2f}%")

    print("\n💡 نحوه لود کردن:")
    print(f"checkpoint = torch.load('{save_path}')")
    print("model = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])")
    print("model.load_state_dict(checkpoint['model_state_dict'])")

except Exception as e:
    print(f"❌ خطا در ذخیره: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("🎉 تمام مراحل با موفقیت انجام شد!")
print("="*70)
