import torch
import torch.nn as nn
from thop import profile
from torchvision.models import resnet50
import ptflops
from ptflops import get_model_complexity_info

# --- بخش ۱: تابع تعریف مدل (بدون تغییر) ---
def get_model():
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

# --- بخش ۲: تابع تحلیل عملکرد برای یک ویدیو با مدت زمان مشخص ---
def analyze_video_cost(video_duration_seconds, fps, image_size=256):
    """
    این تابع هزینه محاسباتی را برای یک ویدیو با مدت زمان مشخص محاسبه می‌کند.
    """
    print(f"--- Analyzing video of {video_duration_seconds} seconds ---")
    
    # 1. ایجاد مدل
    model = get_model()
    model.eval()

    # 2. محاسبه FLOPs برای یک فریم (این مقدار ثابت است)
    input_tensor = torch.randn(1, 3, image_size, image_size)
    flops_per_frame, params = profile(model, inputs=(input_tensor,), verbose=False)

    # 3. محاسبه تعداد کل فریم‌ها و FLOPs کل ویدیو
    total_frames = int(video_duration_seconds * fps)
    flops_per_video = flops_per_frame * total_frames

    # 4. نمایش نتایج
    print(f"  - Total Frames: {total_frames}")
    print(f"  - FLOPs per Frame: {flops_per_frame/1e9:.2f} GFLOPs")
    print(f"  - Total FLOPs for this video: {flops_per_video/1e9:.2f} GFLOPs")
    print("-" * 50)
    
    return flops_per_video

# --- بخش ۳: اجرای اصلی ---
if __name__ == "__main__":
    
    # مشخصات دیتاست ویدیویی شما
    FPS = 30
    MIN_DURATION = 4  # ثانیه
    AVG_DURATION = 11 # ثانیه
    MAX_DURATION = 40 # ثانیه
    IMAGE_SIZE = 256

    print("="*70)
    print("Video Dataset Computational Cost Analysis")
    print("="*70)
    print(f"Dataset Specifications:")
    print(f"  - Frame Rate (FPS): {FPS}")
    print(f"  - Video Duration Range: {MIN_DURATION}s to {MAX_DURATION}s")
    print(f"  - Average Video Duration: {AVG_DURATION}s")
    print(f"  - Input Frame Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("-" * 70)
    
    # محاسبه برای هر سه سناریو
    min_flops = analyze_video_cost(video_duration_seconds=MIN_DURATION, fps=FPS, image_size=IMAGE_SIZE)
    avg_flops = analyze_video_cost(video_duration_seconds=AVG_DURATION, fps=FPS, image_size=IMAGE_SIZE)
    max_flops = analyze_video_cost(video_duration_seconds=MAX_DURATION, fps=FPS, image_size=IMAGE_SIZE)

    # نمایش خلاصه نهایی
    print("\n" + "="*70)
    print("Summary of Computational Cost per Video")
    print("="*70)
    print(f"  - Minimum Cost (4s video): {min_flops/1e9:.2f} GFLOPs")
    print(f"  - Average Cost (11s video): {avg_flops/1e9:.2f} GFLOPs")
    print(f"  - Maximum Cost (40s video): {max_flops/1e9:.2f} GFLOPs")
    print("="*70)

    # --- نکته مهم در مورد پردازش زنده (Real-Time) ---
    print("\n" + "="*70)
    print("Real-Time Processing Requirement (Independent of Video Length)")
    print("="*70)
    # برای پردازش زنده، فقط نرخ فریم مهم است، نه کل طول ویدیو
    # این بخش را می‌توانید مانند قبل محاسبه کنید
    model = get_model()
    model.eval()
    input_tensor = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    flops_per_frame, _ = profile(model, inputs=(input_tensor,), verbose=False)
    
    required_gflops_per_second = (flops_per_frame * FPS) / 1e9
    print(f"To process video at {FPS} FPS, your hardware needs to sustain:")
    print(f"  - Required Computational Power: {required_gflops_per_second:.2f} GFLOP/s")
    print("="*70)
