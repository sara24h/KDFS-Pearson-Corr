arch=MobileNetV2
dataset_dir=/kaggle/input/140k-real-and-fake-faces
dataset_mode=140k
ckpt_path=/kaggle/input/kdfs-3-khordad-fixed-temp/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt
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
