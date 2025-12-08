# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import copy
from thop import profile
import ptflops
from ptflops import get_model_complexity_info

# وارد کردن تابع ایجاد دیتالودر از فایل جداگانه
from dataset import create_uadfv_dataloaders

# --- بخش ۱: تنظیمات و هایپرپارامترها ---
ROOT_DIR = '/kaggle/input/uadfv-dataset/UADFV'  # مسیر دیتاست ویدیویی
MODEL_SAVE_PATH = 'best_resnet50_video_model.pth'
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_EVAL = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_FRAMES = 16
IMAGE_SIZE = 256

# --- بخش ۲: تعریف مدل و فاین تیون ---
def get_model(pretrained=True):
    """
    ایجاد مدل ResNet50 با لایه آخر برای خروجی باینری
    """
    model = models.resnet50(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # خروجی باینری
    return model

def set_parameter_requires_grad(model, feature_extracting):
    """
    منجمد کردن پارامترهای مدل در صورت نیاز (برای فاین تیون پارسیال)
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(feature_extract=False, use_pretrained=True):
    """
    مقداردهی اولیه مدل و مشخص کردن پارامترهایی که باید آموزش ببینند
    """
    model_ft = get_model(use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    
    # فقط پارامترهای لایه fc بهینه‌سازی خواهند شد (اگر feature_extract=True باشد)
    params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
    print(f"Params to learn: {len(params_to_update)}")
    
    return model_ft, params_to_update

# --- بخش ۳: توابع آموزش و اعتبارسنجی ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cpu'):
    """
    حلقه اصلی آموزش و اعتبارسنجی مدل
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    # پردازش ویدیو: ترکیج بچ و زمان
                    B, T, C, H, W = inputs.shape
                    inputs = inputs.view(B * T, C, H, W)
                    labels_repeated = labels.repeat_interleave(T)
                    
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5
                    
                    loss = criterion(outputs, labels_repeated.unsqueeze(1))
                    
                    # بازگرداندن به شکل ویدیویی برای میانگین‌گیری
                    outputs = outputs.view(B, T)
                    preds = preds.view(B, T)
                    
                    # میانگین‌گیری پیش‌بینی‌ها برای هر ویدیو
                    final_preds = preds.mean(dim=1).float()
                    final_labels = labels.squeeze(1).float()

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # آمارها بر اساس پیش‌بینی نهایی ویدیو
                running_loss += loss.item() * inputs.size(0) # loss بر اساس همه فریم‌ها
                running_corrects += torch.sum(final_preds == final_labels)

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / (dataset_size * NUM_FRAMES) # میانگین loss بر فریم
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
    """
    ارزیابی نهایی مدل روی مجموعه تست
    """
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
            final_preds = preds.mean(dim=1).float()
            final_labels = labels.squeeze(1).float()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(final_preds == final_labels)

    test_loss = running_loss / (dataset_size * NUM_FRAMES)
    test_acc = running_corrects.double() / dataset_size
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")

def analyze_model_complexity(model, image_size=256):
    """
    تحلیل پیچیدگی محاسباتی و تعداد پارامترهای مدل
    """
    print("\n" + "="*70)
    print("Model Complexity Analysis after Fine-Tuning")
    print("="*70)
    
    # تحلیل با ptflops برای جزئیات لایه‌ها
    try:
        macs, params = get_model_complexity_info(model, (3, image_size, image_size), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print(f"ResNet FLOPs (ptflops): {macs}")
        print(f"ResNet Parameters (ptflops): {params}")
    except Exception as e:
        print(f"Could not run ptflops analysis: {e}")

    # تحلیل با thop برای خلاصه
    input_tensor = torch.randn(1, 3, image_size, image_size)
    flops, params_thop = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"ResNet MACs (thop): {flops}")
    print(f"ResNet Parameters (thop): {params_thop}")


# --- بخش ۴: اجرای اصلی ---
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ایجاد دیتالودرها با استفاده از تابع از فایل dataset.py
    train_loader, val_loader, test_loader = create_uadfv_dataloaders(
        root_dir=ROOT_DIR, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE,
        train_batch_size=BATCH_SIZE_TRAIN, eval_batch_size=BATCH_SIZE_EVAL,
        num_workers=2 # کاهش تعداد workerها در Kaggle برای جلوگیری از خطا
    )
    dataloaders = {'Train': train_loader, 'Val': val_loader, 'Test': test_loader}

    # مقداردهی اولیه مدل
    model_ft, params_to_update = initialize_model(feature_extract=True, use_pretrained=True)
    model_ft = model_ft.to(device)

    # تعریف تابع هزینه و بهینه‌ساز
    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.Adam(params_to_update, lr=LEARNING_RATE)

    # آموزش مدل
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=NUM_EPOCHS, device=device)
    
    # ذخیره مدل نهایی
    torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved final model at epoch {NUM_EPOCHS}")

    # ارزیابی روی مجموعه تست
    evaluate_model(model_ft, test_loader, criterion, device=device)

    # تحلیل پیچیدگی مدل
    analyze_model_complexity(model_ft, IMAGE_SIZE)
