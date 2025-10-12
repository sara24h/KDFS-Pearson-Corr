import torch
import torch.nn as nn
import sys
sys.path.append('/kaggle/working')

from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

checkpoint_path = '/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt'

print("="*70)
print("تحلیل نوع Pruning و استخراج ماسک‌ها")
print("="*70)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
sparse_state_dict = checkpoint['student']

def analyze_pruning_type(state_dict):
    """
    تشخیص نوع pruning: structured (filter-level) یا unstructured (weight-level)
    """
    print("\n🔍 تحلیل نوع pruning...")
    
    # بررسی چند لایه نمونه
    sample_keys = [k for k in state_dict.keys() if 'conv' in k and 'weight' in k][:5]
    
    structured_pruning = False
    unstructured_pruning = False
    
    for key in sample_keys:
        weight = state_dict[key]
        total_filters = weight.shape[0]
        
        # بررسی آیا فیلتر کامل حذف شده (همه وزن‌های آن صفر)
        filters_all_zero = (weight.view(weight.shape[0], -1).abs().sum(dim=1) == 0).sum().item()
        
        # بررسی آیا وزن‌های individual صفر شدن
        total_weights = weight.numel()
        zero_weights = (weight == 0).sum().item()
        sparsity = zero_weights / total_weights * 100
        
        print(f"\n  {key}:")
        print(f"    - Shape: {list(weight.shape)}")
        print(f"    - فیلترهای کاملاً صفر: {filters_all_zero}/{total_filters}")
        print(f"    - Sparsity: {sparsity:.2f}%")
        
        if filters_all_zero > 0:
            structured_pruning = True
        if sparsity > 5:  # threshold برای تشخیص unstructured pruning
            unstructured_pruning = True
    
    print(f"\n📊 نتیجه تحلیل:")
    print(f"  - Structured Pruning (حذف فیلتر): {'✓' if structured_pruning else '✗'}")
    print(f"  - Unstructured Pruning (صفر کردن وزن): {'✓' if unstructured_pruning else '✗'}")
    
    return structured_pruning, unstructured_pruning

structured, unstructured = analyze_pruning_type(sparse_state_dict)

print("\n" + "="*70)

if not structured:
    print("⚠️  این checkpoint از Structured Pruning استفاده نکرده!")
    print("⚠️  فیلترها حذف نشدن، فقط وزن‌ها sparse شدن.")
    print("\n💡 دو راه‌حل داریم:")
    print("\n1️⃣  استفاده از مدل عادی (بدون pruned architecture):")
    print("   - مدل ResNet50 استاندارد")
    print("   - فقط وزن‌های sparse لود میشن")
    print("   - نیازی به ماسک نیست")
    print("\n2️⃣  تبدیل به Structured Pruning:")
    print("   - فیلترهایی که بیشتر صفر هستن رو حذف کنیم")
    print("   - ماسک بسازیم و مدل pruned استفاده کنیم")
    
    response = input("\n❓ کدوم روش رو میخوای؟ (1 یا 2): ").strip()
    
    if response == "1":
        print("\n" + "="*70)
        print("روش 1: استفاده از مدل استاندارد با وزن‌های Sparse")
        print("="*70)
        
        # استفاده از مدل عادی
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        
        # تغییر fc برای binary classification
        model.fc = nn.Linear(model.fc.in_features, 1)
        
        # لود وزن‌های sparse
        missing, unexpected = model.load_state_dict(sparse_state_dict, strict=False)
        print(f"✅ وزن‌های sparse لود شدند")
        print(f"   - Missing keys: {len(missing)}")
        print(f"   - Unexpected keys: {len(unexpected)}")
        
        # محاسبه sparsity
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        print(f"\n📊 آمار:")
        print(f"   - تعداد کل پارامترها: {total_params:,}")
        print(f"   - پارامترهای صفر: {zero_params:,}")
        print(f"   - Sparsity: {zero_params/total_params*100:.2f}%")
        
        # تست
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 224, 224)
            output = model(dummy_input)
            print(f"\n✅ تست موفق! شکل خروجی: {output.shape}")
        
        # ذخیره
        save_path = '/kaggle/working/resnet50_sparse_weights.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'total_params': total_params,
            'sparsity': zero_params/total_params,
            'model_type': 'standard_resnet50_with_sparse_weights'
        }, save_path)
        
        import os
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n✅ ذخیره شد در: {save_path}")
        print(f"✅ حجم فایل: {file_size_mb:.2f} MB")
        
    else:  # response == "2"
        print("\n" + "="*70)
        print("روش 2: تبدیل به Structured Pruning")
        print("="*70)
        
        def convert_to_structured_pruning(state_dict, threshold=0.7):
            """
            تبدیل unstructured به structured pruning
            فیلترهایی که sparsity بیشتر از threshold دارن رو حذف میکنیم
            """
            masks = []
            pruned_filters = []
            original_filters = []
            
            resnet50_structure = {
                'layer1': {'blocks': 3, 'filters': [64, 64, 256]},
                'layer2': {'blocks': 4, 'filters': [128, 128, 512]},
                'layer3': {'blocks': 6, 'filters': [256, 256, 1024]},
                'layer4': {'blocks': 3, 'filters': [512, 512, 2048]}
            }
            
            print(f"\n🔍 تبدیل با threshold={threshold} (فیلترهایی با >{threshold*100}% sparsity حذف میشن)")
            
            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                num_blocks = resnet50_structure[layer_name]['blocks']
                standard_filters = resnet50_structure[layer_name]['filters']
                
                for block_idx in range(num_blocks):
                    for conv_idx in range(1, 4):
                        conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                        
                        if conv_key in state_dict:
                            weight = state_dict[conv_key]
                            num_filters = weight.shape[0]
                            original_count = standard_filters[conv_idx - 1]
                            
                            # محاسبه sparsity هر فیلتر
                            filter_sparsity = []
                            for i in range(num_filters):
                                filter_weights = weight[i].flatten()
                                zeros = (filter_weights == 0).sum().item()
                                sparsity = zeros / filter_weights.numel()
                                filter_sparsity.append(sparsity)
                            
                            # تعیین فیلترهای باقی‌مانده (sparsity < threshold)
                            active_filters = [i for i, s in enumerate(filter_sparsity) if s < threshold]
                            pruned_count = len(active_filters)
                            
                            # ساخت ماسک
                            mask = torch.zeros(original_count)
                            for i in active_filters:
                                if i < original_count:
                                    mask[i] = 1
                            
                            masks.append(mask)
                            pruned_filters.append(pruned_count)
                            original_filters.append(original_count)
                            
                            avg_sparsity = sum(filter_sparsity) / len(filter_sparsity) * 100
                            print(f"  ✓ {conv_key}: {pruned_count}/{original_count} ({avg_sparsity:.1f}% avg sparsity)")
            
            return masks, pruned_filters, original_filters
        
        masks, pruned_counts, original_counts = convert_to_structured_pruning(sparse_state_dict, threshold=0.7)
        
        print(f"\n✅ تعداد ماسک‌ها: {len(masks)}")
        print(f"📊 نرخ حذف کلی: {(1 - sum(pruned_counts)/sum(original_counts))*100:.2f}%")
        
        # ادامه با ساخت مدل pruned...
        print("\n💡 حالا میتونی از این ماسک‌ها برای ساخت مدل pruned استفاده کنی")

else:
    print("✅ این checkpoint از Structured Pruning استفاده کرده!")
    
    def extract_structured_masks(state_dict):
        """استخراج ماسک‌ها از structured pruning"""
        masks = []
        pruned_filters = []
        original_filters = []
        
        resnet50_structure = {
            'layer1': {'blocks': 3, 'filters': [64, 64, 256]},
            'layer2': {'blocks': 4, 'filters': [128, 128, 512]},
            'layer3': {'blocks': 6, 'filters': [256, 256, 1024]},
            'layer4': {'blocks': 3, 'filters': [512, 512, 2048]}
        }
        
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            num_blocks = resnet50_structure[layer_name]['blocks']
            standard_filters = resnet50_structure[layer_name]['filters']
            
            for block_idx in range(num_blocks):
                for conv_idx in range(1, 4):
                    conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                    
                    if conv_key in state_dict:
                        weight = state_dict[conv_key]
                        
                        # شمارش فیلترهای non-zero
                        filter_norms = weight.view(weight.shape[0], -1).abs().sum(dim=1)
                        active_filters_indices = (filter_norms > 0).nonzero(as_tuple=True)[0]
                        pruned_count = len(active_filters_indices)
                        original_count = standard_filters[conv_idx - 1]
                        
                        # ساخت ماسک
                        mask = torch.zeros(original_count)
                        mask[active_filters_indices] = 1
                        
                        masks.append(mask)
                        pruned_filters.append(pruned_count)
                        original_filters.append(original_count)
                        
                        print(f"  ✓ {conv_key}: {pruned_count}/{original_count} فیلتر")
        
        return masks, pruned_filters, original_filters
    
    masks, pruned_counts, original_counts = extract_structured_masks(sparse_state_dict)
    
    # ادامه با ساخت و لود مدل...
    print(f"\n✅ ماسک‌ها استخراج شدند!")

print("\n" + "="*70)
