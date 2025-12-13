#!/bin/bash

# ========== تنظیمات پیش‌فرض ==========
arch="ResNet_50"
dataset_dir="/kaggle/input/rvf10k"
dataset_mode="rvf10k"      # ✅ تغییر داده شد!
result_dir="/kaggle/working/"
batch_size=256
num_workers=4
device_id=0

# ========== بررسی ورودی‌ها ==========
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <checkpoint_path_1> <checkpoint_path_2>"
    echo "Example:"
    echo "  $0 /kaggle/input/10k-kdfs-seed-2025-data/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt /kaggle/input/10k-pearson-seed5555-data/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt"
    exit 1
fi

ckpt1="$1"
ckpt2="$2"

# ========== بررسی وجود فایل‌ها ==========
if [ ! -f "$ckpt1" ]; then
    echo "❌ Error: Checkpoint 1 not found: $ckpt1"
    exit 1
fi

if [ ! -f "$ckpt2" ]; then
    echo "❌ Error: Checkpoint 2 not found: $ckpt2"
    exit 1
fi

# برای rvf10k، بررسی CSVهای لازم (نه data.csv)
if [ ! -f "/kaggle/input/rvf10k/train.csv" ] || [ ! -f "/kaggle/input/rvf10k/valid.csv" ]; then
    echo "❌ Error: RVF10k CSV files not found in /kaggle/input/rvf10k/"
    exit 1
fi

echo "🚀 Testing two models on RVF10k dataset..."
echo "   Model 1: $ckpt1"
echo "   Model 2: $ckpt2"
echo "   Output: $result_dir"

CUDA_VISIBLE_DEVICES=$device_id python test_comparison.py \
  --dataset_dir "$dataset_dir" \
  --dataset_mode "$dataset_mode" \
  --ckpt1 "$ckpt1" \
  --ckpt2 "$ckpt2" \
  --name1 "KDFS" \
  --name2 "Pearson" \
  --result_dir "$result_dir" \
  --batch_size $batch_size

echo "✅ Done! Results saved in: $result_dir"
