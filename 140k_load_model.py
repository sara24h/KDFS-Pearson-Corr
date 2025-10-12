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

def extract_masks_from_weights(state_dict):
    """
    استخراج ماسک‌ها مستقیماً از وزن‌های واقعی
    فیلترهایی که norm آن‌ها نزدیک صفر است = pruned شده‌اند
    """
    masks = []
    remaining_counts = []
    
    # ساختار ResNet50
    layer_configs = [
        ('layer1', 3),  # 3 blocks
        ('layer2', 4),  # 4 blocks
        ('layer3', 6),  # 6 blocks
        ('layer4', 3),  # 3 blocks
    ]
    
    print("\nاستخراج ماسک‌ها:\n")
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):  # conv1, conv2, conv3
                weight_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                
                if weight_key in state_dict:
                    weight = state_dict[weight_key]
                    num_filters = weight.shape[0]
                    
                    # محاسبه نرم هر فیلتر
                    filter_norms = torch.norm(weight.view(num_filters, -1), p=1, dim=1)
                    
                    # ساخت ماسک: 1 برای فیلترهای فعال، 0 برای pruned
                    mask = (filter_norms >= 1e-6).float()
                    
                    remaining = int(mask.sum().item())
                    
                    masks.append(mask)
                    remaining_counts.append(remaining)
                    
                    print(f"{layer_name}.{block_idx}.conv{conv_idx}: {remaining}/{num_filters} فیلتر")
    
    return masks, remaining_counts

masks, remaining_counts = extract_masks_from_weights(sparse_state_dict)

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
    لود وزن‌های sparse در مدل pruned - نسخه اصلاح شده
    """
    pruned_state_dict = {}
    
    # conv1 و bn1 (بدون تغییر)
    if 'conv1.weight' in sparse_state_dict:
        pruned_state_dict['conv1.weight'] = sparse_state_dict['conv1.weight']
    if 'bn1.weight' in sparse_state_dict:
        pruned_state_dict['bn1.weight'] = sparse_state_dict['bn1.weight']
        pruned_state_dict['bn1.bias'] = sparse_state_dict['bn1.bias']
        pruned_state_dict['bn1.running_mean'] = sparse_state_dict['bn1.running_mean']
        pruned_state_dict['bn1.running_var'] = sparse_state_dict['bn1.running_var']
        pruned_state_dict['bn1.num_batches_tracked'] = sparse_state_dict['bn1.num_batches_tracked']
    
    mask_idx = 0
    layer_configs = [
        ('layer1', 3, 256),   # output channels of layer
        ('layer2', 4, 512),
        ('layer3', 6, 1024),
        ('layer4', 3, 2048),
    ]
    
    print("\nلود وزن‌ها (نسخه اصلاح شده):\n")
    
    for layer_name, num_blocks, layer_out_channels in layer_configs:
        for block_idx in range(num_blocks):
            # برای هر block، input_channels را مشخص کنید
            if block_idx == 0:
                # اولین block: input از layer قبل
                if layer_name == 'layer1':
                    block_in_channels = 64  # از conv1
                else:
                    # از خروجی layer قبل
                    prev_layer_idx = int(layer_name[-1]) - 2
                    prev_out = layer_configs[prev_layer_idx][2]
                    block_in_channels = prev_out
            else:
                # بقیه blockها: input از block قبل همان layer
                block_in_channels = layer_out_channels
            
            for conv_idx in range(1, 4):
                sparse_conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                sparse_bn_key = f'{layer_name}.{block_idx}.bn{conv_idx}.weight'
                
                if sparse_conv_key in sparse_state_dict:
                    sparse_weight = sparse_state_dict[sparse_conv_key]
                    mask = masks[mask_idx]
                    
                    # استخراج فیلترهای فعال
                    active_filters = (mask == 1).nonzero(as_tuple=True)[0]
                    pruned_weight = sparse_weight[active_filters]
                    
                    # تعیین input channels صحیح
                    if conv_idx == 1:
                        # conv1: input از block input
                        expected_in = block_in_channels
                    elif conv_idx == 2:
                        # conv2: input از conv1 (از ماسک قبل)
                        if mask_idx > 0:
                            prev_mask = masks[mask_idx - 1]
                            active_in_channels = (prev_mask == 1).nonzero(as_tuple=True)[0]
                            pruned_weight = pruned_weight[:, active_in_channels]
                    elif conv_idx == 3:
                        # conv3: input از conv2 (از ماسک قبل)
                        if mask_idx > 0:
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
            
            # downsample - نیاز به prune شدن output channels
            downsample_conv_key = f'{layer_name}.{block_idx}.downsample.0.weight'
            if downsample_conv_key in sparse_state_dict:
                downsample_weight = sparse_state_dict[downsample_conv_key]
                
                # output channels باید با conv3 همان block match شود
                conv3_mask = masks[mask_idx - 1]  # ماسک conv3
                active_out = (conv3_mask == 1).nonzero(as_tuple=True)[0]
                
                pruned_downsample = downsample_weight[active_out]
                pruned_state_dict[downsample_conv_key] = pruned_downsample
                
                print(f"{layer_name}.{block_idx}.downsample: {pruned_downsample.shape}")
                
                # BatchNorm downsample
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.weight'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.weight'][active_out]
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.bias'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.bias'][active_out]
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.running_mean'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.running_mean'][active_out]
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.running_var'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.running_var'][active_out]
                pruned_state_dict[f'{layer_name}.{block_idx}.downsample.1.num_batches_tracked'] = \
                    sparse_state_dict[f'{layer_name}.{block_idx}.downsample.1.num_batches_tracked']
    
    # fc - نیاز به prune شدن input features
    if 'fc.weight' in sparse_state_dict:
        fc_weight = sparse_state_dict['fc.weight']
        # input features از آخرین ماسک (layer4.2.conv3)
        last_mask = masks[-1]
        active_features = (last_mask == 1).nonzero(as_tuple=True)[0]
        pruned_fc = fc_weight[:, active_features]
        
        pruned_state_dict['fc.weight'] = pruned_fc
        pruned_state_dict['fc.bias'] = sparse_state_dict['fc.bias']
        print(f"\nfc: {pruned_fc.shape}")
    
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
