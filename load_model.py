import torch
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k

# Determine the appropriate model class based on your dataset
model = ResNet_50_sparse_hardfakevsreal()  # Or ResNet_50_sparse_rvf10k()

# Load the checkpoint
checkpoint_path = '/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract the state dictionary
if 'student' in checkpoint:
    state_dict = checkpoint['student']
else:
    state_dict = checkpoint

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Freeze all layers except the last fully connected layer
for name, param in model.named_parameters():
    if 'fc' not in name:  # unfreeze the last fully connected layer
        param.requires_grad = False

# Set the model to evaluation mode
model.eval()

# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Calculate the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

print("Model loaded successfully!")
