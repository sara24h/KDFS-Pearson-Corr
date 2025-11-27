# فایل: evaluate_final.py

import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm

# ایمپورت ماژول‌های سفارشی خودتان
# مطمئن شوید این مسیرها درست هستند
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_rvf10k, ResNet_50_sparse_hardfakevsreal
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_deepfake
from model.student.GoogleNet_sparse import GoogLeNet_sparse_deepfake

def evaluate(model, data_loader, device, model_name="Model"):
    """یک تابع برای ارزیابی مدل روی یک دیتالودر مشخص."""
    model.eval()
    model.ticket = True  # فعال کردن ماسک نهایی برای ارزیابی
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"Evaluating {model_name}"):
            images = images.to(device)
            targets = targets.to(device).float()
            
            logits, _ = model(images)
            logits = logits.squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    accuracy = 100. * correct / total
    return accuracy

def get_student_model_class(arch, dataset_mode):
    """تابع کمکی برای گرفتن کلاس مدل دانشجو."""
    arch = arch.lower().replace('_', '')
    if arch == 'resnet50':
        return ResNet_50_sparse_rvf10k if dataset_mode != "hardfake" else ResNet_50_sparse_hardfakevsreal
    elif arch == 'mobilenetv2':
        return MobileNetV2_sparse_deepfake
    elif arch == 'googlenet':
        return GoogLeNet_sparse_deepfake
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. ساخت دیتالودرها (با shuffle=False) ---
    print("Creating data loaders with shuffle=False for reliable evaluation...")
    dataset_instance = Dataset_selector(
        dataset_mode=args.dataset_mode,
        # باید تمام پارامترهای لازم را مثل آموزش به آن بدهید
        rvf10k_train_csv=os.path.join(args.dataset_dir, 'train.csv'),
        rvf10k_valid_csv=os.path.join(args.dataset_dir, 'valid.csv'),
        rvf10k_root_dir=args.dataset_dir,
        realfake140k_train_csv=os.path.join(args.dataset_dir, 'train.csv'),
        realfake140k_valid_csv=os.path.join(args.dataset_dir, 'valid.csv'),
        realfake140k_test_csv=os.path.join(args.dataset_dir, 'test.csv'),
        realfake140k_root_dir=args.dataset_dir,
        realfake200k_train_csv="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/train_labels.csv",
        realfake200k_val_csv="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/val_labels.csv",
        realfake200k_test_csv="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/test_labels.csv",
        realfake200k_root_dir=args.dataset_dir,
        realfake190k_root_dir=args.dataset_dir,
        realfake330k_root_dir=args.dataset_dir,
        train_batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        ddp=False # برای ارزیابی نهایی DDP را خاموش می‌کنیم
    )
    val_loader = dataset_instance.loader_val
    test_loader = dataset_instance.loader_test
    print("Data loaders created successfully.")

    # --- 2. بارگذاری مدل‌ها ---
    models_to_evaluate = {}
    
    # بارگذاری بهترین مدل
    best_model_path = "/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt"
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        student_model_class = get_student_model_class(args.arch, args.dataset_mode)
        best_model = student_model_class(
            gumbel_start_temperature=args.gumbel_start_temperature,
            gumbel_end_temperature=args.gumbel_end_temperature,
            num_epochs=args.num_epochs,
        )
        # تنظیم لایه نهایی (کپی شده از کد شما)
        if args.arch == 'mobilenetv2':
            num_ftrs = best_model.classifier.in_features
            best_model.classifier = nn.Linear(num_ftrs, 1)
        elif args.arch == 'googlenet':
            num_ftrs = best_model.fc.in_features
            best_model.fc = nn.Linear(num_ftrs, 1)
        else:
            num_ftrs = best_model.fc.in_features
            best_model.fc = nn.Linear(num_ftrs, 1)
            
        checkpoint = torch.load(best_model_path, map_location='cpu')
        best_model.load_state_dict(checkpoint["student"])
        best_model.to(device)
        models_to_evaluate['Best Model'] = best_model
    else:
        print(f"WARNING: Best model not found at {best_model_path}")

    last_model_path ="/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_last.pt"
    if os.path.exists(last_model_path):
        print(f"Loading last model from: {last_model_path}")
        student_model_class = get_student_model_class(args.arch, args.dataset_mode)
        last_model = student_model_class(
            gumbel_start_temperature=args.gumbel_start_temperature,
            gumbel_end_temperature=args.gumbel_end_temperature,
            num_epochs=args.num_epochs,
        )
        if args.arch == 'mobilenetv2':
            num_ftrs = last_model.classifier.in_features
            last_model.classifier = nn.Linear(num_ftrs, 1)
        elif args.arch == 'googlenet':
            num_ftrs = last_model.fc.in_features
            last_model.fc = nn.Linear(num_ftrs, 1)
        else:
            num_ftrs = last_model.fc.in_features
            last_model.fc = nn.Linear(num_ftrs, 1)

        checkpoint = torch.load(last_model_path, map_location='cpu')
        last_model.load_state_dict(checkpoint["student"])
        last_model.to(device)
        models_to_evaluate['Last Model'] = last_model
    else:
        print(f"WARNING: Last model not found at {last_model_path}")

    if not models_to_evaluate:
        print("ERROR: No models found to evaluate. Exiting.")
        return

    # --- 3. ارزیابی و چاپ نتایج ---
    print("\n--- Starting Final Evaluation ---")
    results = {}

    for model_name, model in models_to_evaluate.items():
        print(f"\n--- Evaluating {model_name} ---")
        
        val_acc = evaluate(model, val_loader, device, model_name)
        test_acc = evaluate(model, test_loader, device, model_name)
        
        results[model_name] = {'val_acc': val_acc, 'test_acc': test_acc}
        print(f"Results for {model_name}:")
        print(f"  -> Reliable Validation Accuracy: {val_acc:.2f}%")
        print(f"  -> Reliable Test Accuracy: {test_acc:.2f}%")

    # --- 4. نتیجه‌گیری نهایی ---
    print("\n--- Final Conclusion ---")
    best_model_name = max(results, key=lambda k: results[k]['val_acc'])
    final_test_acc = results[best_model_name]['test_acc']
    print(f"The best model based on reliable validation accuracy is: {best_model_name}")
    print(f"-> The final, reliable test accuracy to report is: {final_test_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Final Model Evaluation Script")
    # تمام آرگومان‌های لازم را که در آموزش استفاده کردید، اینجا اضافه کنید
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['rvf10k', '140k', '200k', '190k', '330k'])
    parser.add_argument('--result_dir', type=str, required=True, help='Directory where models are saved (e.g., ./results/exp1)')
    parser.add_argument('--arch', type=str, required=True, choices=['resnet50', 'mobilenetv2', 'googlenet'])
    
    # هایپرپارامترهای مدل (باید با آموزش یکسان باشند)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--gumbel_start_temperature', type=float, default=2.0)
    parser.add_argument('--gumbel_end_temperature', type=float, default=0.2)
    
    # تنظیمات دیتالودر
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    main(args)
