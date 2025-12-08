import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights
import time
import copy
from thop import profile
from data.video_data import create_uadfv_dataloaders

ROOT_DIR = '/kaggle/input/uadfv-dataset/UADFV' 
MODEL_SAVE_PATH = 'best_resnet50_video_model.pth'
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_EVAL = 8
NUM_EPOCHS = 6
LEARNING_RATE = 1e-4 

NUM_FRAMES = 32       
IMAGE_SIZE = 256
AVG_VIDEO_SECONDS = 11
VIDEO_FPS = 30

def initialize_model(fine_tune=False, use_pretrained=True):
    model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None)

    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)

    params_to_update = []
    
    if fine_tune:
        print("Setting up Partial Fine-tuning...")
        for param in model_ft.layer3.parameters():
            param.requires_grad = True
            params_to_update.append(param)
        for param in model_ft.layer4.parameters():
            param.requires_grad = True
            params_to_update.append(param)
        print("Unfroze layer3 and layer4.")
    else:
        print("Setting up Feature Extraction (only training the final layer).")

    for param in model_ft.fc.parameters():
        param.requires_grad = True
        params_to_update.append(param)

    total_trainable_params = sum(p.numel() for p in params_to_update)
    print(f"Total trainable parameters: {total_trainable_params:,}")
        
    return model_ft, params_to_update

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
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
                    inputs_flat = inputs.view(B * T, C, H, W)
                    frame_logits = model(inputs_flat)
                    video_logits = frame_logits.view(B, T).mean(dim=1, keepdim=True)
                    loss = criterion(video_logits, labels)
                    preds = (video_logits > 0).float()

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f}, {phase} Accuracy: {epoch_acc:.4f}')

            if phase == 'Val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"Saved best model with validation accuracy: {best_acc:.4f} at epoch {epoch + 1}")
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.6f}')
    
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            B, T, C, H, W = inputs.shape
            inputs_flat = inputs.view(B * T, C, H, W)
            frame_logits = model(inputs_flat)
            video_logits = frame_logits.view(B, T).mean(dim=1, keepdim=True)
            loss = criterion(video_logits, labels)
            preds = (video_logits > 0).float()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

    test_loss = running_loss / dataset_size
    test_acc = running_corrects / dataset_size
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

def analyze_model_complexity(
    model, 
    device, 
    image_size=256, 
    num_frames_sampled=32,   
    avg_video_seconds=11, 
    fps=30
):
    print("\n" + "="*70)
    print("Model Complexity Analysis after Fine-Tuning")
    print("="*70)

    input_tensor = torch.randn(1, 3, image_size, image_size).to(device)

    flops_per_frame, params = profile(model, inputs=(input_tensor,), verbose=False)

    flops_per_video = flops_per_frame * num_frames_sampled
    
    print("\n--- Computational Cost Summary ---")
    print(f"Total Parameters: {params / 1e6:.2f} M")
    print(f"FLOPs per Frame: {flops_per_frame / 1e9:.2f} GFLOPs")
    print(f"Frames Sampled per Video: {num_frames_sampled}")
    print(f"FLOPs per Video (sampled): {flops_per_video / 1e9:.2f} GFLOPs")
    print("-" * 35)

    video_duration_sec = avg_video_seconds
    total_gflops_per_video = flops_per_video / 1e9
    avg_gflops_per_second = total_gflops_per_video / video_duration_sec
    print(f"Average GFLOP/s Required (over {avg_video_seconds}s video): {avg_gflops_per_second:.2f} GFLOP/s")
    print("="*70)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("="*70)
    print("Dataset Configuration")
    print("="*70)
    print(f"Average Video Duration: {AVG_VIDEO_SECONDS} seconds")
    print(f"Video Frame Rate (FPS): {VIDEO_FPS}")
    print(f"Total Frames per Video (avg): {AVG_VIDEO_SECONDS * VIDEO_FPS}")
    print(f"Frames Sampled per Video for Training/Inference: {NUM_FRAMES}")
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

    model_ft, params_to_update = initialize_model(fine_tune=True, use_pretrained=True)
    model_ft = model_ft.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(params_to_update, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=3, verbose=True)

    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, scheduler, num_epochs=NUM_EPOCHS, device=device)
    
    torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved final model at epoch {NUM_EPOCHS}")

    evaluate_model(model_ft, test_loader, criterion, device=device)

    analyze_model_complexity(
        model=model_ft,
        device=device,
        image_size=IMAGE_SIZE,
        num_frames_sampled=NUM_FRAMES,         
        avg_video_seconds=AVG_VIDEO_SECONDS,
        fps=VIDEO_FPS
    )
