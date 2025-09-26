import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import sys
import os
import argparse
from model.student.ResNet_sparse import ResNet_50_sparse_rvf10k
from data.dataset import Dataset_selector

# کلاس هوک برای استخراج activation
class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.activation = None
    def hook_fn(self, module, input, output):
        self.activation = output
    def close(self):
        self.hook.remove()

# تابع برای نمایش نقشه ویژگی‌ها
def visualize_feature_maps(activation, layer_name, num_channels_to_show=16, save_path=None):
    activation = activation.detach().cpu().squeeze(0)
    num_channels = activation.shape[0]
    num_channels_to_show = min(num_channels, num_channels_to_show)
    
    rows = int(np.ceil(num_channels_to_show / 4))
    plt.figure(figsize=(15, 4 * rows))
    
    for i in range(num_channels_to_show):
        plt.subplot(rows, 4, i + 1)
        feature_map = activation[i].numpy()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        plt.imshow(feature_map, cmap='viridis')
        plt.title(f'{layer_name} Channel {i}')
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Feature maps saved as '{save_path}'")
    plt.show()

# تنظیم آرگومان‌های خط فرمان
parser = argparse.ArgumentParser(description='Visualize feature maps for ResNet50 sparse model')
parser.add_argument('--dataset', type=str, default='rvf10k', choices=['rvf10k', 'rvf140k'],
                    help='Dataset to use (rvf10k or rvf140k)')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
parser.add_argument('--image_size', type=int, default=256, help='Input image size')
parser.add_argument('--num_channels', type=int, default=16, help='Number of channels to visualize')
args = parser.parse_args()

# تنظیم مسیرها بر اساس دیتاست
if args.dataset == 'rvf10k':
    rvf_train_csv = '/kaggle/input/rvf10k/train.csv'
    rvf_valid_csv = '/kaggle/input/rvf10k/valid.csv'
    rvf_root_dir = '/kaggle/input/rvf10k'
    default_checkpoint = '/kaggle/input/kdfs-21-mordad-10k-new-pearson-final-data/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt'
elif args.dataset == '140k':
    rvf_train_csv = '/kaggle/input/140k-real-and-fake-faces/train.csv'
    rvf_valid_csv = '/kaggle/input/140k-real-and-fake-faces/valid.csv'
    rvf_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv'
    rvf_root_dir = '/kaggle/input/140k-real-and-fake-faces'
    default_checkpoint = '/kaggle/input/kdfs-140k-r-0-7-27-mordad/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt'  # مسیر واقعی را جایگزین کنید

# استفاده از چک‌پوینت پیش‌فرض یا مشخص‌شده
checkpoint_path = args.checkpoint if args.checkpoint else default_checkpoint

# بررسی وجود فایل‌ها
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
if not os.path.exists(rvf_train_csv) or not os.path.exists(rvf_valid_csv):
    raise FileNotFoundError(f"CSV files for {args.dataset} dataset not found")
if not os.path.exists(rvf_root_dir):
    raise FileNotFoundError(f"{args.dataset} dataset directory not found at {rvf_root_dir}")

# شبیه‌سازی args برای سازگاری با مدل
class ModelArgs:
    def __init__(self):
        self.dataset_mode = args.dataset
        self.gumbel_start_temperature = 5.0
        self.gumbel_end_temperature = 1.0
        self.num_epochs = 100
        self.dataset_type = args.dataset

model_args = ModelArgs()

# لود دیتاست
dataset = Dataset_selector(
    dataset_mode=args.dataset,
    rvf10k_train_csv=rvf_train_csv,
    rvf10k_valid_csv=rvf_valid_csv,
    rvf10k_root_dir=rvf_root_dir,
    train_batch_size=1,
    eval_batch_size=1,
    num_workers=0,
    pin_memory=False,
    ddp=False
)

# گرفتن یک تصویر
val_loader = dataset.loader_val
image, label = next(iter(val_loader))
image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
label = label.item()

# لود مدل
model = ResNet_50_sparse_rvf10k(
    gumbel_start_temperature=model_args.gumbel_start_temperature,
    gumbel_end_temperature=model_args.gumbel_end_temperature,
    num_epochs=model_args.num_epochs,
)
model.dataset_type = model_args.dataset_type
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# لود چک‌پوینت
ckpt = torch.load(checkpoint_path, map_location='cpu')
state_dict = ckpt['student']
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
model.ticket = True

# انتخاب لایه‌ها
first_layer = model.layer1[0].conv1
second_layer = model.layer1[0].conv2

# نمایش تصویر ورودی
image_np = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
image_np = (image_np * np.array([0.2486, 0.2238, 0.2211]) + np.array([0.5212, 0.4260, 0.3811])).clip(0, 1)
plt.figure(figsize=(5, 5))
plt.imshow(image_np)
plt.title(f'Input Image (Label: {"Real" if label == 1 else "Fake"})')
plt.axis('off')
plt.show()

# استخراج و نمایش feature maps
hook_first = Hook(first_layer)
model(image)
vis_first = hook_first.activation
hook_first.close()
visualize_feature_maps(vis_first, 'First_Layer', args.num_channels, save_path=f'feature_maps_first_layer_{args.dataset}.png')

hook_second = Hook(second_layer)
model(image)
vis_second = hook_second.activation
hook_second.close()
visualize_feature_maps(vis_second, 'Second_Layer', args.num_channels, save_path=f'feature_maps_second_layer_{args.dataset}.png')
