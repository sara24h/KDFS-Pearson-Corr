import torch
import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Prune and save ResNet model')
    parser.add_argument('--input', type=str, required=True, help='Input checkpoint path')
    parser.add_argument('--output', type=str, required=True, help='Output checkpoint path')
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_save_path = args.input
    output_save_path = args.output
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        # Load input checkpoint
        checkpoint_loaded = torch.load(input_save_path, map_location=device)

        # Extract key information
        model_state_dict = checkpoint_loaded['model_state_dict']
        masks = checkpoint_loaded['masks']
        
        # Create pruned model using masks
        from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
        model_pruned = ResNet_50_pruned_hardfakevsreal(masks=masks)
        
        # Load pruned weights
        model_pruned.load_state_dict(model_state_dict)
        
        model_pruned = model_pruned.to(device)
        model_pruned.eval()
        
        total_params = sum(p.numel() for p in model_pruned.parameters())
        print("Pruned model successfully reconstructed and loaded!")
        print(f"Number of parameters: {total_params:,}")

        print("\n" + "="*70)
        print("Saving reconstructed model...")
        print("="*70)

        checkpoint_to_save = {
            'model_state_dict': model_pruned.state_dict(),
            'masks': masks,  # Save masks for easy reconstruction in future
            'total_params': total_params,
            'model_architecture': 'ResNet_50_pruned_hardfakevsreal'
        }
        
        torch.save(checkpoint_to_save, output_save_path)
        
        # Print saved file information
        file_size_mb = os.path.getsize(output_save_path) / (1024 * 1024)
        print(f"Model successfully saved at {output_save_path}.")
        print(f"Saved file size: {file_size_mb:.2f} MB")
        
        print("\n" + "="*70)
        print("Final pruned model architecture (ResNet_50_pruned_hardfakevsreal)")
        print("="*70)
        print(model_pruned)
        
    except Exception as e:
        print(f"Error loading or saving model: {e}")

if __name__ == "__main__":
    main()
