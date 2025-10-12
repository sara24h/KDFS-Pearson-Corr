import torch
import torch.nn as nn
import sys
sys.path.append('/kaggle/working')

from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

# مسیر چک‌پوینت جدید
checkpoint_path = '/kaggle/input/kdfs-140k-pearson-19-shahrivar-data/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt'

print("="*70)
print("استخراج خودکار ماسک‌ها از مدل Sparse")
print("="*70)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
sparse_state_dict = checkpoint['student']

def extract_pruned_info_auto(state_dict):
    """
    استخراج خودکار تعداد فیلترهای pruned شده و ساخت ماسک‌ها
    """
    pruned_counts = []
    original_counts = []
    masks = []
    
    # تعداد فیلترهای اصلی ResNet50
    original_filters_per_layer = {
        'layer1': [64, 64, 256] * 3,
        'layer2': [128, 128, 512] * 4,
        'layer3': [256, 256, 1024] * 6,
        'layer4': [512, 512, 2048] * 3,
    }
    
    layer_configs = [
        ('layer1', 3),
        ('layer2', 4),
        ('layer3', 6),
        ('layer4', 3),
    ]
    
    print("\n🔍 در حال بررسی شکل وزن‌ها...")
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):  # conv1, conv2, conv3
                key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                
                if key in state_dict:
                    weight = state_dict[key]
                    num_pruned_filters = weight.shape[0]  # تعداد فیلترهای باقی‌مانده
                    
                    # پیدا کردن تعداد فیلترهای اصلی
                    layer_idx = int(layer_name[-1]) - 1
                    block_conv_idx = block_idx * 3 + (conv_idx - 1)
                    num_original_filters = original_filters_per_layer[layer_name][block_conv_idx]
                    
                    pruned_counts.append(num_pruned_filters)
                    original_counts.append(num_original_filters)
                    
                    # ساخت ماسک
                    mask = torch.zeros(num_original_filters)
                    mask[:num_pruned_filters] = 1
                    masks.append(mask)
                    
                    print(f"  {key:50s} | {num_pruned_filters:4d}/{num_original_filters:4d} ({100*num_pruned_filters/num_original_filters:.1f}%)")
                else:
                    print(f"⚠️  {key} not found!")
    
    return masks, pruned_counts, original_counts

# استخراج اطلاعات
masks, pruned_counts, original_counts = extract_pruned_info_auto(sparse_state_dict)

print(f"\n✅ تعداد ماسک‌های ساخته شده: {len(masks)}")
print(f"✅ میانگین نرخ Pruning: {100*(1 - sum(pruned_counts)/sum(original_counts)):.2f}%")

# آمار کلی
print("\n📊 آمار Pruning:")
print(f"  - پارامترهای اصلی: {sum(original_counts):,}")
print(f"  - پارامترهای باقی‌مانده: {sum(pruned_counts):,}")
print(f"  - کاهش: {sum(original_counts) - sum(pruned_counts):,} پارامتر")

# ===========================
# 2. ساخت مدل Pruned
# ===========================

print("\n" + "="*70)
print("ساخت مدل Pruned")
print("="*70)

try:
    model_pruned = ResNet_50_pruned_hardfakevsreal(masks=masks)
    print("✅ مدل pruned با موفقیت ساخته شد!")
    
    total_params = sum(p.numel() for p in model_pruned.parameters())
    trainable_params = sum(p.numel() for p in model_pruned.parameters() if p.requires_grad)
    
    print(f"✅ تعداد کل پارامترها: {total_params:,}")
    print(f"✅ پارامترهای قابل آموزش: {trainable_params:,}")
    
except Exception as e:
    print(f"❌ خطا در ساخت مدل: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===========================
# 3. لود وزن‌ها (بهبود یافته)
# ===========================

print("\n" + "="*70)
print("لود وزن‌ها در مدل Pruned")
print("="*70)

def load_pruned_weights_improved(model_pruned, sparse_state_dict, masks):
    """
    لود وزن‌های sparse با مدیریت بهتر input/output channels
    """
    pruned_state_dict = {}
    
    # conv1 و bn1
    for key in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 
                'bn1.running_var', 'bn1.num_batches_tracked']:
        if key in sparse_state_dict:
            pruned_state_dict[key] = sparse_state_dict[key]
    
    # fc
    for key in ['fc.weight', 'fc.bias']:
        if key in sparse_state_dict:
            pruned_state_dict[key] = sparse_state_dict[key]
    
    mask_idx = 0
    layer_configs = [
        ('layer1', 3),
        ('layer2', 4),
        ('layer3', 6),
        ('layer4', 3),
    ]
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            # برای هر block، conv1->conv2->conv3
            for conv_idx in range(1, 4):
                sparse_conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                
                if sparse_conv_key not in sparse_state_dict:
                    continue
                
                sparse_weight = sparse_state_dict[sparse_conv_key]
                current_mask = masks[mask_idx]
                
                # فیلترهای فعال در output
                active_out = (current_mask == 1).nonzero(as_tuple=True)[0]
                pruned_weight = sparse_weight[active_out]
                
                # مدیریت input channels
                # conv1: ورودی از conv3 بلاک قبلی یا downsample
                # conv2: ورودی از conv1 همین بلاک
                # conv3: ورودی از conv2 همین بلاک
                
                if conv_idx == 1:
                    # conv1 ورودی از conv3 بلاک قبلی داره
                    if mask_idx > 0:
                        prev_mask = masks[mask_idx - 1]
                        active_in = (prev_mask == 1).nonzero(as_tuple=True)[0]
                        pruned_weight = pruned_weight[:, active_in]
                
                elif conv_idx == 2:
                    # conv2 ورودی از conv1 همین بلاک
                    conv1_mask = masks[mask_idx - 1]
                    active_in = (conv1_mask == 1).nonzero(as_tuple=True)[0]
                    pruned_weight = pruned_weight[:, active_in]
                
                elif conv_idx == 3:
                    # conv3 ورودی از conv2 همین بلاک
                    conv2_mask = masks[mask_idx - 1]
                    active_in = (conv2_mask == 1).nonzero(as_tuple=True)[0]
                    pruned_weight = pruned_weight[:, active_in]
                
                pruned_state_dict[sparse_conv_key] = pruned_weight
                
                # BatchNorm
                bn_keys = [
                    f'{layer_name}.{block_idx}.bn{conv_idx}.weight',
                    f'{layer_name}.{block_idx}.bn{conv_idx}.bias',
                    f'{layer_name}.{block_idx}.bn{conv_idx}.running_mean',
                    f'{layer_name}.{block_idx}.bn{conv_idx}.running_var',
                    f'{layer_name}.{block_idx}.bn{conv_idx}.num_batches_tracked',
                ]
                
                for bn_key in bn_keys:
                    if bn_key in sparse_state_dict:
                        if 'num_batches_tracked' in bn_key:
                            pruned_state_dict[bn_key] = sparse_state_dict[bn_key]
                        else:
                            pruned_state_dict[bn_key] = sparse_state_dict[bn_key][active_out]
                
                mask_idx += 1
            
            # downsample
            downsample_keys = [
                f'{layer_name}.{block_idx}.downsample.0.weight',
                f'{layer_name}.{block_idx}.downsample.1.weight',
                f'{layer_name}.{block_idx}.downsample.1.bias',
                f'{layer_name}.{block_idx}.downsample.1.running_mean',
                f'{layer_name}.{block_idx}.downsample.1.running_var',
                f'{layer_name}.{block_idx}.downsample.1.num_batches_tracked',
            ]
            
            for key in downsample_keys:
                if key in sparse_state_dict:
                    pruned_state_dict[key] = sparse_state_dict[key]
    
    return pruned_state_dict

try:
    print("🔄 در حال تبدیل وزن‌ها...")
    pruned_weights = load_pruned_weights_improved(model_pruned, sparse_state_dict, masks)
    print(f"✅ تبدیل انجام شد: {len(pruned_weights)} کلید")
    
    # لود
    missing, unexpected = model_pruned.load_state_dict(pruned_weights, strict=False)
    print(f"✅ وزن‌ها لود شدند")
    
    if missing:
        print(f"⚠️  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    
    # تست
    print("\n🧪 تست مدل...")
    model_pruned.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224)
        output, features = model_pruned(dummy_input)
        print(f"✅ تست موفق!")
        print(f"   - Output: {output.shape}")
        print(f"   - Features: {len(features)} مپ")
        for i, feat in enumerate(features):
            print(f"      Feature {i}: {feat.shape}")
    
    # ===========================
    # 4. ذخیره
    # ===========================
    
    print("\n" + "="*70)
    print("ذخیره مدل")
    print("="*70)
    
    save_dir = '/kaggle/working'
    
    # چک‌پوینت کامل
    checkpoint_full = {
        'model_state_dict': model_pruned.state_dict(),
        'masks': masks,
        'pruned_counts': pruned_counts,
        'original_counts': original_counts,
        'total_params': total_params,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal',
        'pruning_rate': 1 - sum(pruned_counts)/sum(original_counts),
    }
    
    save_path_full = f'{save_dir}/resnet50_pruned_complete.pt'
    torch.save(checkpoint_full, save_path_full)
    
    # فقط وزن‌ها
    save_path_weights = f'{save_dir}/resnet50_pruned_weights.pt'
    torch.save(model_pruned.state_dict(), save_path_weights)
    
    import os
    size_full = os.path.getsize(save_path_full) / (1024**2)
    size_weights = os.path.getsize(save_path_weights) / (1024**2)
    
    print(f"✅ ذخیره‌سازی کامل شد!")
    print(f"   📦 چک‌پوینت کامل: {save_path_full} ({size_full:.2f} MB)")
    print(f"   📦 فقط وزن‌ها: {save_path_weights} ({size_weights:.2f} MB)")
    
    print("\n💡 نحوه استفاده:")
    print("```python")
    print("checkpoint = torch.load('resnet50_pruned_complete.pt')")
    print("model = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])")
    print("model.load_state_dict(checkpoint['model_state_dict'])")
    print("```")
    
except Exception as e:
    print(f"❌ خطا: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ پایان")
print("="*70)
