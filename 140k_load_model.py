import torch
import torch.nn as nn
import sys
sys.path.append('/kaggle/working')

from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

# ===========================
# مسیر چک‌پوینت جدید
# ===========================
checkpoint_path = '/kaggle/input/kdfs-140k-pearson-19-shahrivar-data/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt'

print("="*70)
print("🔍 بررسی چک‌پوینت و استخراج تعداد فیلترها")
print("="*70)

# لود چک‌پوینت
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# چک کردن کلیدهای موجود
print(f"✅ کلیدهای اصلی چک‌پوینت: {list(checkpoint.keys())}")

# پیدا کردن state_dict
if 'student' in checkpoint:
    sparse_state_dict = checkpoint['student']
    print("✅ state_dict در کلید 'student' پیدا شد")
elif 'model_state_dict' in checkpoint:
    sparse_state_dict = checkpoint['model_state_dict']
    print("✅ state_dict در کلید 'model_state_dict' پیدا شد")
elif 'state_dict' in checkpoint:
    sparse_state_dict = checkpoint['state_dict']
    print("✅ state_dict در کلید 'state_dict' پیدا شد")
else:
    sparse_state_dict = checkpoint
    print("✅ خود checkpoint به عنوان state_dict استفاده می‌شود")

# ===========================
# استخراج خودکار تعداد فیلترها
# ===========================
def auto_extract_pruned_filters(state_dict):
    """
    استخراج خودکار تعداد فیلترهای باقی‌مانده از وزن‌های conv
    """
    pruned_filters = []
    
    layer_configs = [
        ('layer1', 3),  # 3 blocks
        ('layer2', 4),  # 4 blocks
        ('layer3', 6),  # 6 blocks
        ('layer4', 3),  # 3 blocks
    ]
    
    print("\n📊 تعداد فیلترهای باقی‌مانده در هر لایه:")
    print("-" * 70)
    
    for layer_name, num_blocks in layer_configs:
        print(f"\n{layer_name}:")
        for block_idx in range(num_blocks):
            block_filters = []
            for conv_idx in range(1, 4):  # conv1, conv2, conv3
                key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                if key in state_dict:
                    num_filters = state_dict[key].shape[0]
                    block_filters.append(num_filters)
                    pruned_filters.append(num_filters)
            
            if block_filters:
                print(f"  Block {block_idx}: {block_filters[0]:3d}, {block_filters[1]:3d}, {block_filters[2]:3d}")
    
    print("-" * 70)
    print(f"✅ تعداد کل ماسک‌ها: {len(pruned_filters)}")
    
    return pruned_filters

pruned_filters = auto_extract_pruned_filters(sparse_state_dict)

# ===========================
# ساخت ماسک‌ها
# ===========================
print("\n" + "="*70)
print("🎭 ساخت ماسک‌ها")
print("="*70)

def create_masks_from_pruned_filters(pruned_filters):
    """
    ساخت ماسک‌ها بر اساس تعداد فیلترهای pruned شده
    """
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
    
    masks = []
    total_original = 0
    total_pruned = 0
    
    for orig_filters, pruned_count in zip(original_filters, pruned_filters):
        mask = torch.zeros(orig_filters)
        mask[:pruned_count] = 1
        masks.append(mask)
        
        total_original += orig_filters
        total_pruned += pruned_count
    
    sparsity = (1 - total_pruned / total_original) * 100
    
    print(f"✅ تعداد ماسک‌های ساخته شده: {len(masks)}")
    print(f"📊 فیلترهای اصلی: {total_original:,}")
    print(f"📊 فیلترهای باقی‌مانده: {total_pruned:,}")
    print(f"✂️  Sparsity: {sparsity:.2f}%")
    
    return masks, original_filters

masks, original_filters = create_masks_from_pruned_filters(pruned_filters)

# نمایش چند نمونه
print("\n🔍 نمونه ماسک‌ها (5 تای اول):")
for i in range(min(5, len(masks))):
    remaining = int(masks[i].sum())
    total = len(masks[i])
    percentage = (remaining / total) * 100
    print(f"  Mask {i}: {remaining:3d}/{total:4d} ({percentage:5.1f}% باقی‌مانده)")

# ===========================
# ساخت مدل Pruned
# ===========================
print("\n" + "="*70)
print("🏗️  ساخت مدل Pruned")
print("="*70)

try:
    model_pruned = ResNet_50_pruned_hardfakevsreal(masks=masks)
    print("✅ مدل pruned با موفقیت ساخته شد!")
    
    # محاسبه تعداد پارامترها
    total_params = sum(p.numel() for p in model_pruned.parameters())
    trainable_params = sum(p.numel() for p in model_pruned.parameters() if p.requires_grad)
    
    print(f"📊 تعداد کل پارامترها: {total_params:,}")
    print(f"📊 پارامترهای قابل آموزش: {trainable_params:,}")
    
    # مقایسه با ResNet50 استاندارد (تقریباً 25.6M پارامتر)
    standard_params = 25_600_000
    reduction = (1 - total_params / standard_params) * 100
    print(f"📉 کاهش پارامترها نسبت به ResNet50 استاندارد: {reduction:.2f}%")
    
except Exception as e:
    print(f"❌ خطا در ساخت مدل: {e}")
    import traceback
    traceback.print_exc()
    raise

# ===========================
# لود وزن‌ها در مدل Pruned
# ===========================
print("\n" + "="*70)
print("📥 لود وزن‌ها در مدل Pruned")
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
        ('layer1', 3),
        ('layer2', 4),
        ('layer3', 6),
        ('layer4', 3),
    ]
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):  # conv1, conv2, conv3
                sparse_conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                
                if sparse_conv_key in sparse_state_dict:
                    sparse_weight = sparse_state_dict[sparse_conv_key]
                    mask = masks[mask_idx]
                    
                    # استخراج فیلترهای فعال
                    active_filters = (mask == 1).nonzero(as_tuple=True)[0]
                    pruned_weight = sparse_weight[active_filters]
                    
                    # Prune کردن input channels
                    if conv_idx > 1 and mask_idx > 0:
                        prev_mask = masks[mask_idx - 1]
                        active_in_channels = (prev_mask == 1).nonzero(as_tuple=True)[0]
                        pruned_weight = pruned_weight[:, active_in_channels]
                    
                    pruned_state_dict[sparse_conv_key] = pruned_weight
                    
                    # BatchNorm parameters
                    bn_prefix = f'{layer_name}.{block_idx}.bn{conv_idx}'
                    for bn_suffix in ['.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked']:
                        bn_key = bn_prefix + bn_suffix
                        if bn_key in sparse_state_dict:
                            if 'num_batches_tracked' in bn_suffix:
                                pruned_state_dict[bn_key] = sparse_state_dict[bn_key]
                            else:
                                pruned_state_dict[bn_key] = sparse_state_dict[bn_key][active_filters]
                    
                    mask_idx += 1
            
            # downsample
            downsample_prefix = f'{layer_name}.{block_idx}.downsample'
            for ds_key in sparse_state_dict.keys():
                if ds_key.startswith(downsample_prefix):
                    pruned_state_dict[ds_key] = sparse_state_dict[ds_key]
    
    return pruned_state_dict

try:
    pruned_weights = load_pruned_weights(model_pruned, sparse_state_dict, masks)
    print(f"✅ وزن‌های pruned آماده شد: {len(pruned_weights)} کلید")
    
    # لود در مدل
    missing, unexpected = model_pruned.load_state_dict(pruned_weights, strict=False)
    print(f"✅ وزن‌ها با موفقیت لود شدند!")
    
    if missing:
        print(f"⚠️  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    
except Exception as e:
    print(f"❌ خطا در لود وزن‌ها: {e}")
    import traceback
    traceback.print_exc()
    raise

# ===========================
# تست مدل
# ===========================
print("\n" + "="*70)
print("🧪 تست مدل")
print("="*70)

try:
    model_pruned.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224)
        output, features = model_pruned(dummy_input)
        
        print(f"✅ تست موفق!")
        print(f"   📊 شکل ورودی: {dummy_input.shape}")
        print(f"   📊 شکل خروجی: {output.shape}")
        print(f"   📊 تعداد feature maps: {len(features)}")
        print(f"   📊 شکل feature maps: {[f.shape for f in features[:3]]}...")
        
except Exception as e:
    print(f"❌ خطا در تست: {e}")
    import traceback
    traceback.print_exc()
    raise

# ===========================
# ذخیره مدل Pruned
# ===========================
print("\n" + "="*70)
print("💾 ذخیره مدل Pruned")
print("="*70)

try:
    # مسیر ذخیره‌سازی
    save_path = '/kaggle/working/resnet50_pruned_model_140k.pt'
    save_path_weights = '/kaggle/working/resnet50_pruned_weights_only_140k.pt'
    
    # ذخیره کامل (با ماسک‌ها)
    checkpoint_to_save = {
        'model_state_dict': model_pruned.state_dict(),
        'masks': masks,
        'pruned_filters': pruned_filters,
        'original_filters': original_filters,
        'total_params': total_params,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal',
        'source_checkpoint': checkpoint_path
    }
    
    torch.save(checkpoint_to_save, save_path)
    print(f"✅ مدل کامل ذخیره شد: {save_path}")
    
    # ذخیره فقط وزن‌ها
    torch.save(model_pruned.state_dict(), save_path_weights)
    print(f"✅ فقط وزن‌ها ذخیره شد: {save_path_weights}")
    
    # محاسبه حجم فایل‌ها
    import os
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    file_size_weights_mb = os.path.getsize(save_path_weights) / (1024 * 1024)
    
    print(f"\n📦 حجم فایل‌ها:")
    print(f"   - مدل کامل: {file_size_mb:.2f} MB")
    print(f"   - فقط وزن‌ها: {file_size_weights_mb:.2f} MB")
    
    # خلاصه اطلاعات
    print(f"\n📊 خلاصه:")
    print(f"   - تعداد پارامترها: {total_params:,}")
    print(f"   - تعداد ماسک‌ها: {len(masks)}")
    print(f"   - Sparsity: {(1 - sum(pruned_filters) / sum(original_filters)) * 100:.2f}%")
    print(f"   - معماری: ResNet_50_pruned_hardfakevsreal")
    
    print("\n💡 نحوه لود کردن:")
    print("```python")
    print("# لود کامل")
    print(f"checkpoint = torch.load('{save_path}')")
    print("model = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])")
    print("model.load_state_dict(checkpoint['model_state_dict'])")
    print("\n# یا لود سریع‌تر")
    print("model = ResNet_50_pruned_hardfakevsreal(masks=masks)")
    print(f"model.load_state_dict(torch.load('{save_path_weights}'))")
    print("```")
    
    print("\n✅ همه چی تموم شد! مدل آماده استفاده است 🎉")
    
except Exception as e:
    print(f"❌ خطا در ذخیره‌سازی: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
