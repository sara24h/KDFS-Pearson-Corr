
#!/bin/bash

arch=googlenet
dataset_dir=/kaggle/input/hardfakevsrealfaces
dataset_mode=hardfake
ckpt_path=/kaggle/input/kdfs-3-khordad-fixed-temp/results/run_googlenet/student_model/finetune_googlenet_sparse_best.pt
device=0


CUDA_VISIBLE_DEVICES=$device python main.py \
  --phase test \
  --dataset_dir $dataset_dir \
  --dataset_mode $dataset_mode \
  --num_workers 8 \
  --pin_memory \
  --device cuda \
  --arch $arch \
  --test_batch_size 256 \
  --sparsed_student_ckpt_path $ckpt_path \
  "$@"
