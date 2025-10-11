import torch
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_deepfake
from model.pruned_model.MobileNetV2_pruned import MobileNetV2_pruned
from model.student.GoogleNet_sparse import GoogLeNet_sparse_deepfake
from model.pruned_model.GoogleNet_pruned import GoogLeNet_pruned_deepfake
from thop import profile

# Base FLOPs and parameters for each dataset
Flops_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 7700.0,
        "rvf10k": 5390.0,
        "140k": 5390.0,
        "190k": 5390.0,  # Added for 190k
        "200k": 5390.0,
        "330k": 5390.0,
        "125k": 2100.0,
    },
    "MobileNetV2": {
        "hardfakevsrealfaces": 570.0,  # Approximate value for 300x300 input
        "rvf10k": 416.68,
        "140k": 416.68,
        "200k": 416.68,
        "330k": 416.68,
        "190k": 416.68,
        "125k": 153.0,  # Approximate for 160x160 input
    },
    "googlenet": {
        "hardfakevsrealfaces": 570.0,  # Approximate value for 300x300 input
        "rvf10k": 1980,
        "140k": 1980,
        "200k": 1980,
        "330k": 1980,
        "190k": 1980,
        "125k": 153.0,  # Approximate for 160x160 input
    }
}
Params_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 14.97,
        "rvf10k": 23.51,
        "140k": 23.51,
        "190k": 23.51,  # Added for 190k
        "200k": 23.51,
        "330k": 23.51,
        "125k": 23.51,
    },
    "MobileNetV2": {
        "hardfakevsrealfaces": 2.23,
        "rvf10k": 2.23,
        "140k": 2.23,
        "200k": 2.23,
        "330k": 2.23,
        "190k": 2.23,
        "125k": 2.23,
    },
    "googlenet": {
        "hardfakevsrealfaces": 2.23,
        "rvf10k": 5.6,
        "140k": 5.6,
        "200k": 5.6,
        "330k": 5.6,
        "190k": 5.6,
        "125k": 2.23,
    }
}
image_sizes = {
    "hardfakevsreal": 300,
    "rvf10k": 256,
    "140k": 256,
    "190k": 256,  # Added for 190k
    "200k": 256,
    "330k": 256,
    "125k": 160,
}

def get_flops_and_params(dataset_mode, sparsed_student_ckpt_path):
    # Map dataset_mode to dataset_type
    dataset_type = {
        "hardfake": "hardfakevsreal",
        "rvf10k": "rvf10k",
        "140k": "140k",
        "190k": "190k", 
        "200k": "200k",
        "330k": "330k",
        "125k": "125k"
    }[dataset_mode]

    # Load checkpoint
    ckpt_student = torch.load(sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt_student["student"]

    model_type = "ResNet_50"
    student = ResNet_50_sparse_hardfakevsreal()
        

    student.load_state_dict(state_dict)


    # Extract masks
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]

    # Load pruned model with masks

    pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)

    
    # Set input size based on dataset
    input = torch.rand([1, 3, image_sizes[dataset_type], image_sizes[dataset_type]])
    Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)

    # Use dataset-specific baseline values
    Flops_baseline = Flops_baselines[model_type][dataset_type]
    Params_baseline = Params_baselines[model_type][dataset_type]

    Flops_reduction = (
        (Flops_baseline - Flops / (10**6)) / Flops_baseline * 100.0
    )
    Params_reduction = (
        (Params_baseline - Params / (10**6)) / Params_baseline * 100.0
    )
    return (
        Flops_baseline,
        Flops / (10**6),
        Flops_reduction,
        Params_baseline,
        Params / (10**6),
        Params_reduction,
    )

def main():
    # مسیر فایل چک‌پوینت مدل
    sparsed_student_ckpt_path = "/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt"

    # بررسی وجود مسیر
    import os
    if not os.path.exists(sparsed_student_ckpt_path):
        raise ValueError(f"Checkpoint path {sparsed_student_ckpt_path} does not exist.")

    # ارزیابی برای تمام دیتاست‌ها
    for dataset_mode in ["hardfake"]:
        print(f"\nEvaluating for dataset: {dataset_mode}")
        (
            Flops_baseline,
            Flops,
            Flops_reduction,
            Params_baseline,
            Params,
            Params_reduction,
        ) = get_flops_and_params(dataset_mode, sparsed_student_ckpt_path)
        print(
            "Params_baseline: %.2fM, Params: %.2fM, Params reduction: %.2f%%"
            % (Params_baseline, Params, Params_reduction)
        )
        print(
            "Flops_baseline: %.2fM, Flops: %.2fM, Flops reduction: %.2f%%"
            % (Flops_baseline, Flops, Flops_reduction)
        )

if __name__ == "__main__":
    main()
