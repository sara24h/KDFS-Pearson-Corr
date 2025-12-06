import torch
import torch.nn as nn
from thop import profile
from torchvision.models import resnet50
from model.teacher.ResNet import ResNet_50_hardfakevsreal

def calculate_flops_params_for_video(model_path, num_frames=16, image_size=256):
   
    print(f"Loading model from: {model_path}")

    model = ResNet_50_hardfakevsreal()
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'student' in checkpoint: 
             state_dict = checkpoint['student']
        else:
            state_dict = checkpoint 

        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # حذف 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
            
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Please ensure the model architecture in 'get_model()' matches the checkpoint.")
        return

    model.eval() # تنظیم مدل در حالت ارزیابی

    # 3. ایجاد ورودی جعلی برای thop
    # ورودی برای ResNet50 باید یک فریم باشد
    input_tensor = torch.randn(1, 3, image_size, image_size)

    # 4. محاسبه FLOPs و پارامترها برای یک فریم
    flops_per_frame, params = profile(model, inputs=(input_tensor,), verbose=False)

    # 5. محاسبه FLOPs برای کل ویدیو
    flops_per_video = flops_per_frame * num_frames

    # 6. نمایش نتایج
    print("\n" + "="*70)
    print("Calculation Results for Video Model")
    print("="*70)
    print(f"Model Configuration:")
    print(f"  - Architecture: ResNet50 (for binary classification)")
    print(f"  - Input Frame Size: {image_size}x{image_size}")
    print(f"  - Number of Frames per Video: {num_frames}")
    print("-" * 70)
    print(f"Model Metrics:")
    print(f"  - Total Parameters: {params/1e6:.2f} M")
    print(f"  - FLOPs per Frame: {flops_per_frame/1e9:.2f} GFLOPs")
    print(f"  - FLOPs per Video: {flops_per_video/1e9:.2f} GFLOPs")
    print("="*70)

if __name__ == "__main__":

    TEACHER_MODEL_PATH = "/kaggle/input/10k_teacher_beaet/pytorch/default/1/10k-teacher_model_best.pth"

    NUM_FRAMES = 16
    IMAGE_SIZE = 256

    # اجرای تابع محاسبه
    calculate_flops_params_for_video(
        model_path=TEACHER_MODEL_PATH,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE
    )
