import torch
import os
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

input_save_path = '/kaggle/working/resnet50_base_pruned_model_learnable_masks.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_save_path = '/kaggle/working/200k_base_final.pt'

try:
    checkpoint_loaded = torch.load(input_save_path, map_location=device)

    model_state_dict = checkpoint_loaded['model_state_dict']
    masks = checkpoint_loaded['masks']

    model_pruned = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    model_pruned.load_state_dict(model_state_dict)
    
    model_pruned = model_pruned.to(device)
    model_pruned.eval()
    
    total_params = sum(p.numel() for p in model_pruned.parameters())
    print("The pruned model was successfully reconstructed and loaded.")
    print(f"Total parameters: {total_params:,}")

    print("\n" + "="*70)
    print("Saving the reconstructed model...")
    print("="*70)

    checkpoint_to_save = {
        'model_state_dict': model_pruned.state_dict(),
        'masks': masks,  
        'total_params': total_params,
        'model_architecture': 'ResNet_50_pruned_hardfakevsreal'
    }
    
    torch.save(checkpoint_to_save, output_save_path)

    file_size_mb = os.path.getsize(output_save_path) / (1024 * 1024)
    print(f"The model was successfully saved at {output_save_path}.")
    print(f"Saved file size: {file_size_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("Final pruned model architecture (ResNet_50_pruned_hardfakevsreal)")
    print("="*70)
    print(model_pruned)
    
except Exception as e:
    print(f"Error while loading or saving the model: {e}")
