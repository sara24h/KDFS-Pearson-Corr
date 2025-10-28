import torch
import torch.nn as nn
import sys
sys.path.append('/kaggle/working')

from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal

checkpoint_path = '/kaggle/input/330k-08-finetune-30-mordad/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt'

print("="*70)
print("Step 1: Load Student Sparse model")
print("="*70)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
sparse_state_dict = checkpoint['student']

student = ResNet_50_sparse_hardfakevsreal()
student.load_state_dict(sparse_state_dict)
print("Student Sparse model loaded successfully.")

print("\n" + "="*70)
print("Step 2: Extract masks from Learnable Mask Parameters (Method 2)")
print("="*70)

def extract_masks_from_student(student_model):
    masks = []
    remaining_counts = []
    
    print("\nExtracting masks:\n")

    if hasattr(student_model, 'mask_modules'):
        mask_modules = student_model.mask_modules
        print(f"Number of mask modules: {len(mask_modules)}\n")
        
        for idx, mask_module in enumerate(mask_modules):
            if hasattr(mask_module, 'mask_weight'):
                mask_weight = mask_module.mask_weight  # shape: (num_filters, 2, 1, 1)
                
                mask = torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
                
                num_filters = mask.shape[0]
                remaining = int(mask.sum().item())
                
                masks.append(mask)
                remaining_counts.append(remaining)

                if idx < 36:  # Show detailed info for first layers
                    if idx % 3 == 0:
                        block_num = idx // 3
                        if idx < 9:
                            layer_name = f"layer1.{block_num}"
                        elif idx < 21:
                            layer_name = f"layer2.{(idx-9)//3}"
                        elif idx < 39:
                            layer_name = f"layer3.{(idx-21)//3}"
                        else:
                            layer_name = f"layer4.{(idx-39)//3}"
                        print(f"{layer_name}:")
                    
                    conv_num = (idx % 3) + 1
                    print(f"  conv{conv_num}: {remaining}/{num_filters} filters (Binary mask)")
    else:
        print("Model does not have mask_modules.")
        return None, None
    
    return masks, remaining_counts

masks, remaining_counts = extract_masks_from_student(student)

if masks is not None:
    print(f"\nNumber of generated masks: {len(masks)}")
    print(f"Total remaining filters: {sum(remaining_counts)}")
    
    print("\n" + "="*70)
    print("Remaining filters:")
    print("="*70)
    print("remaining_filters = [")
    
    layer_names = []
    for i in range(1, 5):
        num_blocks = [3, 4, 6, 3][i-1]
        for j in range(num_blocks):
            layer_names.append(f'layer{i}.{j}')
    
    idx = 0
    for layer_name in layer_names:
        if idx < len(remaining_counts):
            layer_data = remaining_counts[idx:idx+3]
            print(f"    # {layer_name}")
            print(f"    {', '.join(map(str, layer_data))},")
            idx += 3
    print("]")
else:
    print("Error in extracting masks.")
    raise RuntimeError("Cannot extract masks from student model")

print("\n" + "="*70)
print("Step 3: Compare mask extraction methods")
print("="*70)

def extract_masks_from_weights(state_dict):
    masks_old = []
    remaining_counts_old = []
    
    layer_configs = [
        ('layer1', 3),
        ('layer2', 4),
        ('layer3', 6),
        ('layer4', 3),
    ]
    
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
                    
                    masks_old.append(mask)
                    remaining_counts_old.append(remaining)
    
    return masks_old, remaining_counts_old

masks_old, remaining_counts_old = extract_masks_from_weights(sparse_state_dict)

print("\nComparison of results:\n")
print(f"Method 1 (Norm-based):")
print(f"  - Number of masks: {len(masks_old)}")
print(f"  - Total remaining filters: {sum(remaining_counts_old)}")

print(f"\nMethod 2 (Learnable Mask-based):")
print(f"  - Number of masks: {len(masks)}")
print(f"  - Total remaining filters: {sum(remaining_counts)}")

print(f"\nDifference:")
print(f"  - Difference in filters: {sum(remaining_counts) - sum(remaining_counts_old)} filters")
print(f"  - Percentage difference: {((sum(remaining_counts) - sum(remaining_counts_old)) / sum(remaining_counts_old) * 100):.2f}%")

print("\n" + "="*70)
print("Step 4: Build the Pruned model")
print("="*70)

try:
    model_pruned = ResNet_50_pruned_hardfakevsreal(masks=masks)
    print("Pruned model built successfully.")
    
    total_params = sum(p.numel() for p in model_pruned.parameters())
    print(f"Number of parameters in pruned model: {total_params:,}")
    
except Exception as e:
    print(f"Error in building pruned model: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "="*70)
print("Step 5: Load weights into Pruned model")
print("="*70)

def load_pruned_weights(model_pruned, sparse_state_dict, masks):
    pruned_state_dict = {}

    if 'conv1.weight' in sparse_state_dict:
        pruned_state_dict['conv1.weight'] = sparse_state_dict['conv1.weight']
    if 'bn1.weight' in sparse_state_dict:
        pruned_state_dict['bn1.weight'] = sparse_state_dict['bn1.weight']
        pruned_state_dict['bn1.bias'] = sparse_state_dict['bn1.bias']
        pruned_state_dict['bn1.running_mean'] = sparse_state_dict['bn1.running_mean']
        pruned_state_dict['bn1.running_var'] = sparse_state_dict['bn1.running_var']
        pruned_state_dict['bn1.num_batches_tracked'] = sparse_state_dict['bn1.num_batches_tracked']
    
    if 'fc.weight' in sparse_state_dict:
        pruned_state_dict['fc.weight'] = sparse_state_dict['fc.weight']
        pruned_state_dict['fc.bias'] = sparse_state_dict['fc.bias']
    
    mask_idx = 0
    layer_configs = [
        ('layer1', 3),
        ('layer2', 4),
        ('layer3', 6),
        ('layer4', 3),
    ]
    
    print("\nLoading weights:\n")
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            for conv_idx in range(1, 4):
                sparse_conv_key = f'{layer_name}.{block_idx}.conv{conv_idx}.weight'
                sparse_bn_key = f'{layer_name}.{block_idx}.bn{conv_idx}.weight'
                
                if sparse_conv_key in sparse_state_dict:
                    sparse_weight = sparse_state_dict[sparse_conv_key]
                    mask = masks[mask_idx]
                    
                    active_filters = (mask == 1).nonzero(as_tuple=True)[0]

                    pruned_weight = sparse_weight[active_filters]

                    if conv_idx > 1 and mask_idx > 0:
                        prev_mask = masks[mask_idx - 1]
                        active_in_channels = (prev_mask == 1).nonzero(as_tuple=True)[0]
                        pruned_weight = pruned_weight[:, active_in_channels]
                    
                    pruned_state_dict[sparse_conv_key] = pruned_weight
                    
                    print(f"{layer_name}.{block_idx}.conv{conv_idx}: {pruned_weight.shape}")
                    
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
    print(f"\nPruned weights ready: {len(pruned_weights)} keys")

    missing, unexpected = model_pruned.load_state_dict(pruned_weights, strict=False)
    print(f"Weights loaded successfully.")
    print(f"   - Missing keys: {len(missing)}")
    print(f"   - Unexpected keys: {len(unexpected)}")
    
    if len(missing) > 0:
        print(f"   - Example missing keys: {missing[:3]}")
    if len(unexpected) > 0:
        print(f"   - Example unexpected keys: {unexpected[:3]}")
    
except Exception as e:
    print(f"Error loading weights: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "="*70)
print("Step 6: Test the Pruned model")
print("="*70)

try:
    model_pruned.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224)
        output, features = model_pruned(dummy_input)
        print(f"Test successful.")
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Number of feature maps: {len(features)}")
        if len(features) > 0:
            print(f"   - First feature map shape: {features[0].shape}")
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "="*70)
print("Step 7: Save the Pruned model")
print("="*70)

try:
    save_path = '/kaggle/working/resnet50_base_pruned_model_learnable_masks.pt'
    
    checkpoint_to_save = {
        'model_state_dict': model_pruned.state_dict(),
        'masks': masks,
        'remaining_counts': remaining_counts,
        'total_params': total_params,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal',
        'method': 'learnable_masks',
        'best_prec1': checkpoint.get('best_prec1_after_finetune', None)
    }
    
    torch.save(checkpoint_to_save, save_path)
    print(f"Full model saved at: {save_path}")
    
    import os
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    save_path_weights = '/kaggle/working/resnet50_base_pruned_weights_learnable_masks.pt'
    torch.save(model_pruned.state_dict(), save_path_weights)
    file_size_weights_mb = os.path.getsize(save_path_weights) / (1024 * 1024)
    print(f"Weights only saved at: {save_path_weights}")
    print(f"File size (weights only): {file_size_weights_mb:.2f} MB")
    
    print("\nSaved information:")
    print(f"   - Number of parameters: {total_params:,}")
    print(f"   - Number of masks: {len(masks)}")
    print(f"   - Total remaining filters: {sum(remaining_counts)}")
    print(f"   - Extraction method: Learnable Mask Parameters (Method 2)")
    if checkpoint_to_save['best_prec1']:
        print(f"   - Best accuracy: {checkpoint_to_save['best_prec1']:.2f}%")
    
except Exception as e:
    print(f"Error during save: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("All steps completed successfully.")
print("="*70)
