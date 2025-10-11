import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
from thop import profile

# --- Sparse Model Definitions ---

class SoftMaskedConv2d(nn.Conv2d):  # Assuming this is defined elsewhere, but for completeness
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add mask_weight, etc., as per your implementation
        self.mask_weight = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))  # Placeholder

    def forward(self, x, ticket):
        # Placeholder for masked conv forward
        return super().forward(x)

class MaskedNet(nn.Module):
    def __init__(self, gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200):
        super().__init__()
        self.gumbel_start_temperature = gumbel_start_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.num_epochs = num_epochs
        self.gumbel_temperature = gumbel_start_temperature
        self.ticket = False
        self.mask_modules = []

    def checkpoint(self):
        for m in self.mask_modules:
            m.checkpoint()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules:
            m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)

    def update_gumbel_temperature(self, epoch):
        self.gumbel_temperature = self.gumbel_start_temperature * math.pow(
            self.gumbel_end_temperature / self.gumbel_start_temperature,
            epoch / self.num_epochs,
        )
        for m in self.mask_modules:
            m.update_gumbel_temperature(self.gumbel_temperature)

    def get_flops(self):
        device = next(self.parameters()).device
        Flops_total = torch.tensor(0.0, device=device)
        image_sizes = {
            "hardfakevsrealfaces": 300,
            "rvf10k": 256,
            "140k": 256  # اضافه کردن اندازه تصویر برای دیتاست 140k
        }
        dataset_type = getattr(self, "dataset_type", "hardfakevsrealfaces")
        input_size = image_sizes.get(dataset_type, 256)  # مقدار پیش‌فرض 256 در صورت عدم وجود
        
        conv1_h = (input_size - 7 + 2 * 3) // 2 + 1
        maxpool_h = (conv1_h - 3 + 2 * 1) // 2 + 1
        conv1_w = conv1_h
        maxpool_w = maxpool_h
        
        Flops_total = Flops_total + (
            conv1_h * conv1_w * 7 * 7 * 3 * 64 +
            conv1_h * conv1_w * 64
        )
        
        for i, m in enumerate(self.mask_modules):
            m = m.to(device)
            Flops_shortcut_conv = 0
            Flops_shortcut_bn = 0
            if len(self.mask_modules) == 48:  # برای ResNet-50
                if i % 3 == 0:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        m.in_channels * m.mask.sum()
                    )
                else:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        self.mask_modules[i - 1].mask.to(device).sum() * m.mask.sum()
                    )
                Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
                if i % 3 == 2:
                    Flops_shortcut_conv = (
                        m.feature_map_h * m.feature_map_w * 1 * 1 *
                        (m.out_channels // 4) * m.out_channels
                    )
                    Flops_shortcut_bn = m.feature_map_h * m.feature_map_w * m.out_channels
            elif len(self.mask_modules) in [16, 32]:  # برای مدل‌های دیگر
                if i % 2 == 0:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        m.in_channels * m.mask.sum()
                    )
                else:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        self.mask_modules[i - 1].mask.to(device).sum() * m.mask.sum()
                    )
                Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
                if i % 2 == 1 and i != 1:
                    Flops_shortcut_conv = (
                        m.feature_map_h * m.feature_map_w * 1 * 1 *
                        m.out_channels * m.out_channels
                    )
                    Flops_shortcut_bn = m.feature_map_h * m.feature_map_w * m.out_channels

            Flops_total = (
                Flops_total + Flops_conv + Flops_bn + Flops_shortcut_conv + Flops_shortcut_bn
            )
        return Flops_total

class BasicBlock_sparse(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SoftMaskedConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

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
        out = self.bn2(self.conv2(out, ticket))
        out += self.downsample(x)
        out = F.relu(out)
        return out

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
        num_classes=1,  # تغییر به 1 برای خروجی باینری
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
        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if block == BasicBlock_sparse:
            expansion = 1
        elif block == Bottleneck_sparse:
            expansion = 4
        self.feat1 = nn.Conv2d(64 * expansion, 64 * expansion, kernel_size=1)
        self.feat2 = nn.Conv2d(128 * expansion, 128 * expansion, kernel_size=1)
        self.feat3 = nn.Conv2d(256 * expansion, 256 * expansion, kernel_size=1)
        self.feat4 = nn.Conv2d(512 * expansion, 512 * expansion, kernel_size=1)

        self.mask_modules = [m for m in self.modules() if isinstance(m, SoftMaskedConv2d)]
        self.mask_modules = [m.to(next(self.parameters()).device) for m in self.mask_modules]

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
        feature_list.append(self.feat1(out))

        for block in self.layer2:
            out = block(out, self.ticket)
        feature_list.append(self.feat2(out))

        for block in self.layer3:
            out = block(out, self.ticket)
        feature_list.append(self.feat3(out))

        for block in self.layer4:
            out = block(out, self.ticket)
        feature_list.append(self.feat4(out))

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
        num_classes=1,  # تغییر به 1 برای خروجی باینری
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
        dataset_type="hardfakevsrealfaces"
    )

# --- Pruned Model Definitions ---

def get_preserved_filter_num(mask):
    return int(mask.sum())

class BasicBlock_pruned(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, masks=[], stride=1):
        super().__init__()
        self.masks = masks

        preserved_filter_num1 = get_preserved_filter_num(masks[0])
        self.conv1 = nn.Conv2d(
            in_planes,
            preserved_filter_num1,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(preserved_filter_num1)
        preserved_filter_num2 = get_preserved_filter_num(masks[1])
        self.conv2 = nn.Conv2d(
            preserved_filter_num1,
            preserved_filter_num2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(preserved_filter_num2)

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
        out = self.bn2(self.conv2(out))

        # padding 0 for feature map to get the same shape of short cut
        shortcut_out = self.downsample(x).clone()
        padded_out = torch.zeros_like(shortcut_out).clone()
        for padded_feature_map, feature_map in zip(padded_out, out):
            padded_feature_map[self.masks[1] == 1] = feature_map

        assert padded_out.shape == shortcut_out.shape, "wrong shape"

        padded_out += shortcut_out
        padded_out = F.relu(padded_out)
        return padded_out

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

        # padding 0 for feature map to get the same shape of short cut
        shortcut_out = self.downsample(x).clone()
        padded_out = torch.zeros_like(shortcut_out).clone()
        for padded_feature_map, feature_map in zip(padded_out, out):
            padded_feature_map[self.masks[2] == 1] = feature_map

        assert padded_out.shape == shortcut_out.shape, "wrong shape"

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

        coef = 0
        if block == BasicBlock_pruned:
            coef = 2
        elif block == Bottleneck_pruned:
            coef = 3
        num = 0
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            masks=masks[0 : coef * num_blocks[0]],
        )
        num = num + coef * num_blocks[0]

        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            masks=masks[num : num + coef * num_blocks[1]],
        )
        num = num + coef * num_blocks[1]

        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            masks=masks[num : num + coef * num_blocks[2]],
        )
        num = num + coef * num_blocks[2]

        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            masks=masks[num : num + coef * num_blocks[3]],
        )
        num = num + coef * num_blocks[3]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, masks=[]):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        coef = 0
        if block == BasicBlock_pruned:
            coef = 2
        elif block == Bottleneck_pruned:
            coef = 3

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

# --- Functions ---

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

def prune_and_load_weights(sparsed_student_ckpt_path, dataset_mode="rvf10k"):
    # نقشه دیتاست به نوع
    dataset_type = {
        "hardfake": "hardfakevsreal",
        "rvf10k": "rvf10k",
        "140k": "140k",
        "190k": "190k",
        "200k": "200k",
        "330k": "330k",
        "125k": "125k"
    }[dataset_mode]

    # بارگذاری چک‌پوینت مدل اسپارس
    ckpt_student = torch.load(sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt_student["student"]

    # ایجاد مدل اسپارس و بارگذاری وزن‌ها
    student = ResNet_50_sparse_hardfakevsreal()
    student.load_state_dict(state_dict)
    student.eval()  # برای inference

    # استخراج ماسک‌ها
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]

    # ایجاد مدل هرس‌شده با ماسک‌ها
    pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    pruned_model.eval()

    # انتقال وزن‌های حفظ‌شده از مدل اسپارس به مدل pruned
    # این کار لایه به لایه انجام می‌شود (برای ResNet-50 با Bottleneck)
    sparse_modules = list(student.modules())  # لیست تمام ماژول‌های اسپارس
    pruned_modules = list(pruned_model.modules())  # لیست تمام ماژول‌های pruned

    mask_idx = 0  # ایندکس برای ماسک‌ها
    sparse_idx = 0
    for pruned_module in pruned_modules:
        if isinstance(pruned_module, nn.Conv2d):
            # پیدا کردن لایه conv مربوطه در اسپارس
            while not isinstance(sparse_modules[sparse_idx], SoftMaskedConv2d):
                sparse_idx += 1
            sparse_conv = sparse_modules[sparse_idx]

            # ماسک برای خروجی (out_mask) و ورودی (in_mask)
            out_mask = masks[mask_idx]
            in_mask = masks[mask_idx - 1] if mask_idx > 0 else torch.ones(sparse_conv.in_channels)

            # کپی وزن‌ها
            pruned_module.weight.data = sparse_conv.weight.data[out_mask.bool()][:, in_mask.bool(), :, :].clone()
            if pruned_module.bias is not None:
                pruned_module.bias.data = sparse_conv.bias.data[out_mask.bool()].clone()

            mask_idx += 1
            sparse_idx += 1

        elif isinstance(pruned_module, nn.BatchNorm2d):
            # پیدا کردن BN مربوطه
            while not isinstance(sparse_modules[sparse_idx], nn.BatchNorm2d):
                sparse_idx += 1
            sparse_bn = sparse_modules[sparse_idx]

            # ماسک برای BN (بر اساس ماسک خروجی لایه conv قبلی)
            bn_mask = masks[mask_idx - 1]

            pruned_module.weight.data = sparse_bn.weight.data[bn_mask.bool()].clone()
            pruned_module.bias.data = sparse_bn.bias.data[bn_mask.bool()].clone()
            pruned_module.running_mean = sparse_bn.running_mean[bn_mask.bool()].clone()
            pruned_module.running_var = sparse_bn.running_var[bn_mask.bool()].clone()

            sparse_idx += 1

    # برای لایه‌های دیگر مانند conv1 و fc، مستقیم کپی کنید (بدون پرونینگ فرض شده)
    pruned_model.conv1.weight.data = student.conv1.weight.data.clone()
    pruned_model.bn1.weight.data = student.bn1.weight.data.clone()
    pruned_model.bn1.bias.data = student.bn1.bias.data.clone()
    pruned_model.bn1.running_mean = student.bn1.running_mean.clone()
    pruned_model.bn1.running_var = student.bn1.running_var.clone()

    pruned_model.fc.weight.data = student.fc.weight.data.clone()
    pruned_model.fc.bias.data = student.fc.bias.data.clone()

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

    # بارگذاری مدل هرس‌شده
    pruned_model = prune_and_load_weights(sparsed_student_ckpt_path, dataset_mode="rvf10k")

    # مثال استفاده
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pruned_model.to(device)
    
    input_size = image_sizes["rvf10k"]
    dummy_input = torch.rand(1, 3, input_size, input_size).to(device)
    output, features = pruned_model(dummy_input)
    print("Output shape:", output.shape)

    # ذخیره مدل
    torch.save(pruned_model.state_dict(), "pruned_resnet50_rvf10k.pt")
    print("مدل هرس‌شده ذخیره شد.")

if __name__ == "__main__":
    main()
