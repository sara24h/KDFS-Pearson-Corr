import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from thop import profile
import os
import warnings
warnings.filterwarnings("ignore")

# تنظیم دستگاه محاسباتی
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# لود مستقیم مدل از فایل .pt
try:
    model = torch.load('/kaggle/input/pruned_resnet50_140k/pytorch/default/1/pruned_model (3).pt', map_location=device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise Exception("Failed to load the model.")
model = model.to(device)
model.eval()

# تأیید تعداد پارامترها و FLOPs
input = torch.randn(1, 3, 224, 224).to(device)
flops, params = profile(model, inputs=(input,))
print(f"Params: {params/1e6:.2f}M, FLOPs: {flops/1e6:.2f}M")

# تنظیمات دیتاست‌ها
dataset_configs = {
    'hardfake': {
        'dataset_mode': 'hardfake',
        'hardfake_csv_file': '/kaggle/input/hardfakevsrealfaces/data.csv',
        'hardfake_root_dir': '/kaggle/input/hardfakevsrealfaces',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    'rvf10k': {
        'dataset_mode': 'rvf10k',
        'rvf10k_train_csv': '/kaggle/input/rvf10k/train.csv',
        'rvf10k_valid_csv': '/kaggle/input/rvf10k/valid.csv',
        'rvf10k_root_dir': '/kaggle/input/rvf10k',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '140k': {
        'dataset_mode': '140k',
        'realfake140k_train_csv': '/kaggle/input/140k-real-and-fake-faces/train.csv',
        'realfake140k_valid_csv': '/kaggle/input/140k-real-and-fake-faces/valid.csv',
        'realfake140k_test_csv': '/kaggle/input/140k-real-and-fake-faces/test.csv',
        'realfake140k_root_dir': '/kaggle/input/140k-real-and-fake-faces',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '190k': {
        'dataset_mode': '190k',
        'realfake190k_root_dir': '/kaggle/input/deepfake-and-real-images/Dataset',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '200k': {
        'dataset_mode': '200k',
        'realfake200k_train_csv': '/kaggle/input/200k-real-and-fake-faces/train_labels.csv',
        'realfake200k_val_csv': '/kaggle/input/200k-real-and-fake-faces/val_labels.csv',
        'realfake200k_test_csv': '/kaggle/input/200k-real-and-fake-faces/test_labels.csv',
        'realfake200k_root_dir': '/kaggle/input/200k-real-and-fake-faces',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    },
    '330k': {
        'dataset_mode': '330k',
        'realfake330k_root_dir': '/kaggle/input/deepfake-dataset',
        'train_batch_size': 64,
        'eval_batch_size': 64,
        'ddp': False
    }
}

# بارگذاری دیتاست‌ها
from data.dataset import Dataset_selector
datasets = {}
for dataset_name, config in dataset_configs.items():
    try:
        datasets[dataset_name] = Dataset_selector(**config)
        print(f"{dataset_name} dataset loaded successfully")
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")

# تابع ارزیابی
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions) * 100
    f1 = f1_score(true_labels, predictions, average='binary') * 100
    avg_loss = running_loss / len(dataloader)
    return accuracy, f1, avg_loss

# تست بدون فاین‌تیون
criterion = nn.BCEWithLogitsLoss()
results_without_finetune = {}

for dataset_name, dataset in datasets.items():
    print(f"\nTesting {dataset_name} without fine-tuning...")
    accuracy, f1, loss = evaluate(model, dataset.loader_test, criterion)
    results_without_finetune[dataset_name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'loss': loss
    }
    print(f"{dataset_name} Without Fine-tuning - Accuracy: {accuracy:.2f}%, F1-Score: {f1:.2f}%, Loss: {loss:.4f}")

# تابع آموزش برای فاین‌تیون
def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(trainloader):.4f}")
        
        # ارزیابی روی داده اعتبارسنجی
        accuracy, f1, val_loss = evaluate(model, valloader, criterion)
        print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy:.2f}%, F1-Score: {f1:.2f}%, Validation Loss: {val_loss:.4f}")

# فاین‌تیون برای هر دیتاست
results_with_finetune = {}

for dataset_name, dataset in datasets.items():
    print(f"\nFine-tuning on {dataset_name}...")
    
    # لود مجدد مدل برای هر دیتاست
    model = torch.load('/kaggle/input/pruned_resnet50_140k/pytorch/default/1/pruned_model (3).pt', map_location=device)
    model = model.to(device)
    
    # فریز کردن لایه‌های اولیه
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # آموزش
    train_model(model, dataset.loader_train, dataset.loader_val, criterion, optimizer, num_epochs=5)
    
    # تست روی داده تست
    accuracy, f1, loss = evaluate(model, dataset.loader_test, criterion)
    results_with_finetune[dataset_name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'loss': loss
    }
    print(f"{dataset_name} With Fine-tuning - Accuracy: {accuracy:.2f}%, F1-Score: {f1:.2f}%, Loss: {loss:.4f}")
    
    # ذخیره مدل فاین‌تیون‌شده
    torch.save(model, f'finetuned_{dataset_name}.pt')

# گزارش نتایج
print("\nSummary of Results:")
print("\nWithout Fine-tuning:")
for dataset_name, metrics in results_without_finetune.items():
    print(f"{dataset_name}: Accuracy = {metrics['accuracy']:.2f}%, F1-Score = {metrics['f1_score']:.2f}%, Loss = {metrics['loss']:.4f}")

print("\nWith Fine-tuning:")
for dataset_name, metrics in results_with_finetune.items():
    print(f"{dataset_name}: Accuracy = {metrics['accuracy']:.2f}%, F1-Score = {metrics['f1_score']:.2f}%, Loss = {metrics['loss']:.4f}")
