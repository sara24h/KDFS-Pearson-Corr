import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
from thop import profile

# --- SoftMaskedConv2d Definition ---

class SoftMaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.mask_weight = nn.Parameter(torch.empty(out_channels, 2, 1, 1).normal_(mean=0.0, std=0.01))
        self.gumbel_temperature = 2.0
        self.feature_map_h = 0  # Placeholder
        self.feature_map_w = 0
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask = None

    def forward(self, x, ticket):
        if ticket:
            self.mask = (torch.argmax(self.mask_weight, dim=1) == 1).float().unsqueeze(1).unsqueeze(1).unsqueeze(0)
        else:
            logits = self.mask_weight
            soft_mask = F.gumbel_softmax(logits, tau=self.gumbel_temperature, hard=False, dim=1)
            self.mask = soft_mask[:, 1:2, :, :]
        out = super().forward(x)
        return out * self.mask

# --- Sparse Model Definitions ---

class MaskedNet(nn.Module):
    def __init__(self, gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200):
        super().__init__()
        self.gumbel_start_temperature = gumbel_start_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.num_epochs = num_epochs
        self.gumbel_temperature = gumbel_start_temperature
        self.ticket = False
        self.mask_modules = []

    def update_gumbel_temperature(self, epoch):
        self.gumbel_temperature = self.gumbel_start_temperature * math.pow(
            self.gumbel_end_temperature / self.gumbel_start_temperature,
            epoch / self.num_epochs,
        )
        for m in self.mask_modules:
            m.gumbel_temperature = self.gumbel_temperature

class Bottleneck_sparse(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SoftMaskedConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SoftMaskedConv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, ticket):
        out = F.relu(self.bn1(self.conv1(x, ticket)))
        out = F.relu(self.bn2(self.conv2(out, ticket)))
        out = self.bn3(self.conv3(out, ticket))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet_sparse(MaskedNet):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=1,
        gumbel_start_temperature=2.0,
        gumbel_end_temperature=0.5,
        num_epochs=200,
        dataset_type="hardfakevsrealfaces"
    ):
        super().__init__(
            gumbel_start_temperature,
            gumbel_end_temperature,
            num_epochs,
        )
        self.in_planes = 64
        self.dataset_type = dataset_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.mask_modules = [m for m in self.modules() if isinstance(m, SoftMaskedConv2d)]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        for block in self.layer1:
            out = block(out, self.ticket)
        feature_list.append(out)

        for block in self.layer2:
            out = block(out, self.ticket)
        feature_list.append(out)

        for block in self.layer3:
            out = block(out, self.ticket)
        feature_list.append(out)

        for block in self.layer4:
            out = block(out, self.ticket)
        feature_list.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list

def ResNet_50_sparse_hardfakevsreal(
    gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200
):
    return ResNet_sparse(
        block=Bottleneck_sparse,
        num_blocks=[3, 4, 6, 3],
        num_classes=1,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
        dataset_type="hardfakevsrealfaces"
    )

# --- Pruned Model Definitions ---

def get_preserved_filter_num(mask):
    return int(mask.sum())

class Bottleneck_pruned(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, masks=[], stride=1):
        super().__init__()
        self.masks = masks

        preserved_filter_num1 = get_preserved_filter_num(masks[0])
        self.conv1 = nn.Conv2d(
            in_planes, preserved_filter_num1, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(preserved_filter_num1)
        preserved_filter_num2 = get_preserved_filter_num(masks[1])
        self.conv2 = nn.Conv2d(
            preserved_filter_num1,
            preserved_filter_num2,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(preserved_filter_num2)
        preserved_filter_num3 = get_preserved_filter_num(masks[2])
        self.conv3 = nn.Conv2d(
            preserved_filter_num2,
            preserved_filter_num3,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(preserved_filter_num3)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        shortcut_out = self.downsample(x)
        padded_out = torch.zeros_like(shortcut_out)

        # اصلاح‌شده: استفاده از indexing مستقیم روی کانال‌ها برای جلوگیری از in-place روی view
        indices = torch.nonzero(self.masks[2] == 1).squeeze(-1)  # ایندکس‌های کانال‌هایی که ماسک 1 دارن
        padded_out[:, indices, :, :] = out

        padded_out += shortcut_out
        padded_out = F.relu(padded_out)
        return padded_out

class ResNet_pruned(nn.Module):
    def __init__(self, block, num_blocks, masks=[], num_classes=1):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        coef = 3 if block == Bottleneck_pruned else 2
        num = 0
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            masks=masks[0 : coef * num_blocks[0]],
        )
        num += coef * num_blocks[0]

        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            masks=masks[num : num + coef * num_blocks[1]],
        )
        num += coef * num_blocks[1]

        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            masks=masks[num : num + coef * num_blocks[2]],
        )
        num += coef * num_blocks[2]

        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            masks=masks[num : num + coef * num_blocks[3]],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, masks=[]):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        coef = 3 if block == Bottleneck_pruned else 2
        for i, stride in enumerate(strides):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    masks[coef * i : coef * i + coef],
                    stride,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        for block in self.layer1:
            out = block(out)
        feature_list.append(out)

        for block in self.layer2:
            out = block(out)
        feature_list.append(out)

        for block in self.layer3:
            out = block(out)
        feature_list.append(out)

        for block in self.layer4:
            out = block(out)
        feature_list.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list

def ResNet_50_pruned_hardfakevsreal(masks):
    return ResNet_pruned(
        block=Bottleneck_pruned, num_blocks=[3, 4, 6, 3], masks=masks, num_classes=1
    )

# --- Baselines and Image Sizes ---

Flops_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 7700.0,
        "rvf10k": 5390.0,
        "140k": 5390.0,
        "190k": 5390.0,
        "200k": 5390.0,
        "330k": 5390.0,
        "125k": 2100.0,
    },
}
Params_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 14.97,
        "rvf10k": 23.51,
        "140k": 23.51,
        "190k": 23.51,
        "200k": 23.51,
        "330k": 23.51,
        "125k": 23.51,
    },
}
image_sizes = {
    "hardfakevsreal": 300,
    "rvf10k": 256,
    "140k": 256,
    "190k": 256,
    "200k": 256,
    "330k": 256,
    "125k": 160,
}

# --- Functions ---

def get_flops_and_params(dataset_mode, sparsed_student_ckpt_path):
    dataset_type = {
        "hardfake": "hardfakevsreal",
        "rvf10k": "rvf10k",
        "140k": "140k",
        "190k": "190k",
        "200k": "200k",
        "330k": "330k",
        "125k": "125k"
    }[dataset_mode]

    ckpt_student = torch.load(sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt_student["student"]

    model_type = "ResNet_50"
    student = ResNet_50_sparse_hardfakevsreal()
    student.load_state_dict(state_dict, strict=False)

    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]

    pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    input = torch.rand([1, 3, image_sizes[dataset_type], image_sizes[dataset_type]])
    Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)

    Flops_baseline = Flops_baselines[model_type][dataset_type]
    Params_baseline = Params_baselines[model_type][dataset_type]

    Flops_reduction = ((Flops_baseline - Flops / (10**6)) / Flops_baseline * 100.0)
    Params_reduction = ((Params_baseline - Params / (10**6)) / Params_baseline * 100.0)
    return (
        Flops_baseline,
        Flops / (10**6),
        Flops_reduction,
        Params_baseline,
        Params / (10**6),
        Params_reduction,
    )

def prune_and_load_weights(sparsed_student_ckpt_path, dataset_mode="rvf10k"):
    dataset_type = {
        "hardfake": "hardfakevsreal",
        "rvf10k": "rvf10k",
        "140k": "140k",
        "190k": "190k",
        "200k": "200k",
        "330k": "330k",
        "125k": "125k"
    }[dataset_mode]

    ckpt_student = torch.load(sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt_student["student"]

    student = ResNet_50_sparse_hardfakevsreal()
    student.load_state_dict(state_dict, strict=False)
    student.eval()

    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]

    expected_mask_count = 3 * (3 + 4 + 6 + 3)  # 48 masks for ResNet-50
    if len(masks) != expected_mask_count:
        raise ValueError(f"Expected {expected_mask_count} masks, got {len(masks)}")

    # Debug: Print mask shapes
    for i, mask in enumerate(masks):
        print(f"Mask {i} shape: {mask.shape}")

    pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    pruned_model.eval()

    # Get all Conv2d and SoftMaskedConv2d modules
    sparse_modules = [m for m in student.modules() if isinstance(m, (SoftMaskedConv2d, nn.Conv2d))]
    pruned_modules = [m for m in pruned_model.modules() if isinstance(m, nn.Conv2d)]

    # Filter out non-pruned Conv2d layers (conv1 and downsample layers)
    pruned_conv_modules = []
    for module in pruned_modules:
        if module is pruned_model.conv1:
            continue
        is_downsample = False
        for layer in [pruned_model.layer1, pruned_model.layer2, pruned_model.layer3, pruned_model.layer4]:
            for block in layer:
                if module is getattr(block.downsample, '0', None):
                    is_downsample = True
                    break
            if is_downsample:
                break
        if not is_downsample:
            pruned_conv_modules.append(module)

    if len(masks) != len(pruned_conv_modules):
        raise ValueError(f"Expected {len(pruned_conv_modules)} masks for Conv2d layers, got {len(masks)}")

    # Copy weights for pruned Conv2d layers
    mask_idx = 0
    sparse_conv_idx = 0
    for pruned_module in pruned_conv_modules:
        while sparse_conv_idx < len(sparse_modules) and not isinstance(sparse_modules[sparse_conv_idx], SoftMaskedConv2d):
            sparse_conv_idx += 1
        if sparse_conv_idx >= len(sparse_modules):
            raise ValueError(f"Ran out of SoftMaskedConv2d modules at mask_idx {mask_idx}")
        sparse_module = sparse_modules[sparse_conv_idx]

        out_mask = masks[mask_idx].bool()
        in_mask = masks[mask_idx - 1].bool() if mask_idx > 0 else torch.ones(sparse_module.in_channels, dtype=torch.bool)

        print(f"Processing Conv2d at mask_idx {mask_idx}")
        print(f"  Sparse weight shape: {sparse_module.weight.shape}")
        print(f"  Pruned weight shape: {pruned_module.weight.shape}")
        print(f"  Out mask shape: {out_mask.shape}, In mask shape: {in_mask.shape}")

        if out_mask.shape[0] != sparse_module.weight.shape[0]:
            raise ValueError(f"Out mask shape {out_mask.shape} does not match weight shape {sparse_module.weight.shape} at mask_idx {mask_idx}")
        if in_mask.shape[0] != sparse_module.weight.shape[1]:
            raise ValueError(f"In mask shape {in_mask.shape} does not match weight shape {sparse_module.weight.shape} at mask_idx {mask_idx}")

        try:
            pruned_module.weight.data = sparse_module.weight.data[out_mask][:, in_mask, :, :].clone()
        except IndexError as e:
            raise IndexError(f"Shape mismatch in Conv2d at mask_idx {mask_idx}: {e}")

        mask_idx += 1
        sparse_conv_idx += 1

    # Copy BatchNorm layers
    sparse_bn_modules = []
    pruned_bn_modules = []
    bn_mask_indices = []
    bn_mask_idx = 0  # Separate index for BN masks, starting from 0

    # Collect BatchNorm2d layers in the correct order
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        sparse_layer = getattr(student, layer_name)
        pruned_layer = getattr(pruned_model, layer_name)
        for block_idx, (sparse_block, pruned_block) in enumerate(zip(sparse_layer, pruned_layer)):
            # Collect bn1, bn2, bn3 for each block
            sparse_bn_modules.extend([sparse_block.bn1, sparse_block.bn2, sparse_block.bn3])
            pruned_bn_modules.extend([pruned_block.bn1, pruned_block.bn2, pruned_block.bn3])
            # Assign mask indices sequentially (0,1,2 for first block, 3,4,5 for second, etc.)
            bn_mask_indices.extend([bn_mask_idx, bn_mask_idx + 1, bn_mask_idx + 2])
            bn_mask_idx += 3

    if len(pruned_bn_modules) != len(masks):
        raise ValueError(f"Expected {len(pruned_bn_modules)} masks for BatchNorm2d layers, got {len(masks)}")

    # Debug: Print BatchNorm2d layers and their mask indices
    for i, (sparse_bn, pruned_bn, mask_idx) in enumerate(zip(sparse_bn_modules, pruned_bn_modules, bn_mask_indices)):
        print(f"Processing BatchNorm2d at index {i} (mask_idx {mask_idx})")
        print(f"  Sparse BN features: {sparse_bn.num_features}")
        print(f"  Pruned BN features: {pruned_bn.num_features}")
        print(f"  BN mask shape: {masks[mask_idx].shape}")

        bn_mask = masks[mask_idx].bool()

        if bn_mask.shape[0] != sparse_bn.num_features:
            raise ValueError(f"BN mask shape {bn_mask.shape} does not match sparse BN features {sparse_bn.num_features} at mask_idx {mask_idx}")

        try:
            pruned_bn.weight.data = sparse_bn.weight.data[bn_mask].clone()
            pruned_bn.bias.data = sparse_bn.bias.data[bn_mask].clone()
            pruned_bn.running_mean = sparse_bn.running_mean[bn_mask].clone()
            pruned_bn.running_var = sparse_bn.running_var[bn_mask].clone()
        except IndexError as e:
            raise IndexError(f"Shape mismatch in BatchNorm2d at mask_idx {mask_idx}: {e}")

    # Copy non-pruned layers (conv1, bn1, downsample, fc)
    pruned_model.conv1.weight.data = student.conv1.weight.data.clone()
    pruned_model.bn1.weight.data = student.bn1.weight.data.clone()
    pruned_model.bn1.bias.data = student.bn1.bias.data.clone()
    pruned_model.bn1.running_mean = student.bn1.running_mean.clone()
    pruned_model.bn1.running_var = student.bn1.running_var.clone()

    pruned_model.fc.weight.data = student.fc.weight.data.clone()
    pruned_model.fc.bias.data = student.fc.bias.data.clone()

    # Copy downsample layers
    for layer_name in ['layer2', 'layer3', 'layer4']:
        sparse_layer = getattr(student, layer_name)
        pruned_layer = getattr(pruned_model, layer_name)
        for i in range(len(sparse_layer)):
            if len(sparse_layer[i].downsample) > 0:
                pruned_layer[i].downsample[0].weight.data = sparse_layer[i].downsample[0].weight.data.clone()
                pruned_layer[i].downsample[1].weight.data = sparse_layer[i].downsample[1].weight.data.clone()
                pruned_layer[i].downsample[1].bias.data = sparse_layer[i].downsample[1].bias.data.clone()
                pruned_layer[i].downsample[1].running_mean = sparse_layer[i].downsample[1].running_mean.clone()
                pruned_layer[i].downsample[1].running_var = sparse_layer[i].downsample[1].running_var.clone()

    return pruned_model

def main():
    sparsed_student_ckpt_path = "/kaggle/input/kdfs-10k-pearson-19-shahrivar-314-epochs/results/run_resnet50_imagenet_prune1/student_model/finetune_ResNet_50_sparse_best.pt"
    
    if not os.path.exists(sparsed_student_ckpt_path):
        raise ValueError(f"Checkpoint path {sparsed_student_ckpt_path} does not exist.")

    print("\nEvaluating flops and params for dataset: rvf10k")
    (
        Flops_baseline,
        Flops,
        Flops_reduction,
        Params_baseline,
        Params,
        Params_reduction,
    ) = get_flops_and_params("rvf10k", sparsed_student_ckpt_path)
    print(
        "Params_baseline: %.2fM, Params: %.2fM, Params reduction: %.2f%%"
        % (Params_baseline, Params, Params_reduction)
    )
    print(
        "Flops_baseline: %.2fM, Flops: %.2fM, Flops reduction: %.2f%%"
        % (Flops_baseline, Flops, Flops_reduction)
    )

    pruned_model = prune_and_load_weights(sparsed_student_ckpt_path, dataset_mode="rvf10k")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pruned_model.to(device)
    
    input_size = image_sizes["rvf10k"]
    dummy_input = torch.rand(1, 3, input_size, input_size).to(device)
    output, features = pruned_model(dummy_input)
    print("Output shape:", output.shape)

    torch.save(pruned_model.state_dict(), "pruned_resnet50_rvf10k.pt")
    print("مدل هرس‌شده ذخیره شد.")

if __name__ == "__main__":
    main()
