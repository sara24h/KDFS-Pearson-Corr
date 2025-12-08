# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights
import time
import copy
from thop import profile
import ptflops
from ptflops import get_model_complexity_info

# وارد کردن تابع ایجاد دیتالودر از فایل جداگانه
from data.video_data import create_uadfv_dataloaders

# --- بخش ۱: تنظیمات و هایپرپارامترها ---
ROOT_DIR = '/kaggle/input/uadfv-dataset/UADFV'  # مسیر دیتاست ویدیویی
MODEL_SAVE_PATH = 'best_resnet50_video_model.pth'
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_EVAL = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

# --- هایپرپارامترهای جدید بر اساس اطلاعات دیتاست ---
NUM_FRAMES = 32
IMAGE_SIZE = 256
AVG_VIDEO_SECONDS = 11
VIDEO_FPS = 30

# --- بخش ۲: تعریف مدل و فاین تیون ---
def initialize_model(feature_extract=False, use_pretrained=True):
    model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
    if feature_extract:
        for param in model_ft.parameters():
            param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
    print(f"Params to learn: {len(params_to_update)}")
    if len(params_to_update) == 0:
        print("Warning: No parameters to learn. Check the feature_extract flag.")
    return model_ft, params_to_update

# --- بخش ۳: توابع آموزش و اعتبارسنجی ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cpu'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    B, T, C, H, W = inputs.shape
                    inputs = inputs.view(B * T, C, H, W)
                    labels_repeated = labels.repeat_interleave(T)
                    
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5
                    
                    loss = criterion(outputs, labels_repeated.unsqueeze(1))
                    
                    outputs = outputs.view(B, T)
                    preds = preds.view(B, T)
                    
                    # --- اصلاح شده: تبدیل به float قبل از mean ---
                    final_preds = preds.float().mean(dim=1)
                    final_labels = labels.squeeze(1).float()

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(final_preds > 0.5).float() # مقایسه نهایی نیز باید با float انجام شود

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / (dataset_size * NUM_FRAMES)
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f}, {phase} Accuracy: {epoch_acc:.2%}')

            if phase == 'Val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"Saved best model with validation accuracy: {best_acc:.2%} at epoch {epoch + 1}")
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    dataset_size = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            B, T, C, H, W = inputs.shape
            inputs = inputs.view(B * T, C, H, W)
            labels_repeated = labels.repeat_interleave(T)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels_repeated.unsqueeze(1))
            
            outputs = outputs.view(B, T)
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.view(B, T)
            
            # --- اصلاح شده: تبدیل به float قبل از mean ---
            final_preds = preds.float().mean(dim=1)
            final_labels = labels.squeeze(1).float()
        
        running_loss += loss.item() * inputs.size(0)
        # مقایسه نهایی نیز باید با float انجام شود
        running_corrects += torch.sum(final_preds > 0.5).float()

    test_loss = running_loss / (dataset_size * NUM_FRAMES)
    test_acc = running_corrects.double() / dataset_size
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")

def analyze_model_complexity(model, image_size=256, avg_video_seconds=11, fps=30):
    print("\n" + "="*70)
    print("Model Complexity Analysis after Fine-Tuning")
    print("="*70)
    
    input_tensor = torch.randn(1, 3, image_size, image_size)
    flops_per_frame, params_thop = profile(model, inputs=(input_tensor,), verbose=False)
    
    avg_frames_per_video = int(avg_video_seconds * fps)
    flops_per_avg_video = flops_per_frame * avg_frames_per_video
    
    print("\n--- Computational Cost Summary ---")
    print(f"Total Parameters: {params_thop/1e6:.2f} M")
    print(f"FLOPs per Frame: {flops_per_frame / 1e9:.2f} GFLOPs")
    print(f"Average Frames per Video ({avg_video_seconds}s @ {fps}fps): {avg_frames_per_video}")
    print(f"FLOPs per Average Video: {flops_per_avg_video / 1e9:.2f} GFLOPs")
    print("-" * 35)
    
    required_gflops_per_second = (flops_per_frame * fps) / 1e9
    print(f"Required Hardware Power for Real-Time Processing ({fps} FPS): {required_gflops_per_second:.2f} GFLOP/s")
    print("="*70)


# --- بخش ۴: اجرای اصلی ---
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("="*70)
    print("Dataset Configuration")
    print("="*70)
    print(f"Average Video Duration: {AVG_VIDEO_SECONDS} seconds")
    print(f"Video Frame Rate (FPS): {VIDEO_FPS}")
    print(f"Average Frames per Video: {AVG_VIDEO_SECONDS * VIDEO_FPS}")
    print(f"Frames Sampled per Video for Training: {NUM_FRAMES}")
    print("="*70)

    train_loader, val_loader, test_loader = create_uadfv_dataloaders(
        root_dir=ROOT_DIR, 
        num_frames=NUM_FRAMES, 
        image_size=IMAGE_SIZE,
        train_batch_size=BATCH_SIZE_TRAIN, 
        eval_batch_size=BATCH_SIZE_EVAL,
        num_workers=2
    )
    dataloaders = {'Train': train_loader, 'Val': val_loader, 'Test': test_loader}

    model_ft, params_to_update = initialize_model(feature_extract=True, use_pretrained=True)
    model_ft = model_ft.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(params_to_update, lr=LEARNING_RATE)

    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=NUM_EPOCHS, device=device)
    
    torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved final model at epoch {NUM_EPOCHS}")

    evaluate_model(model_ft, test_loader, criterion, device=device)

    analyze_model_complexity(model_ft, IMAGE_SIZE, AVG_VIDEO_SECONDS, VIDEO_FPS)
