import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from thop import profile
from .layer import SoftMaskedConv2d

class MaskedNet(nn.Module):
    def __init__(self, gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200, num_frames=16):
        super().__init__()
        self.gumbel_start_temperature = gumbel_start_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.num_epochs = num_epochs
        self.gumbel_temperature = gumbel_start_temperature
        self.ticket = False
        self.mask_modules = []
        # --- تغییر جدید: اضافه کردن تعداد فریم‌ها برای محاسبه FLOPs ویدیویی ---
        self.num_frames = num_frames

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
   
        image_sizes = {
            "hardfakevsrealfaces": 300,
            "rvf10k": 256,
            "140k": 256,
            "uadfv": 256
        }
        dataset_type = getattr(self, "dataset_type", "hardfakevsrealfaces")
        input_size = image_sizes.get(dataset_type, 256)
    
    # --- شروع تغییر ---
    # دستگاه مدل را دریافت کنید (GPU یا CPU)
        device = next(self.parameters()).device
    
    # یک تنسور ورودی ساختگی برای یک فریم روی همان دستگاه مدل بسازید
        dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    # --- پایان تغییر ---
    
    # استفاده از کتابخانه thop برای محاسبه دقیق FLOPs برای یک فریم
        flops_per_frame, _ = profile(self, inputs=(dummy_input,), verbose=False)

    # ضرب FLOPs در تعداد فریم‌ها برای به دست آوردن FLOPs کل ویدیو
        total_flops = flops_per_frame * self.num_frames
    
        return total_flops

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
        num_classes=1,
        gumbel_start_temperature=2.0,
        gumbel_end_temperature=0.5,
        num_epochs=200,
        dataset_type="hardfakevsrealfaces",
        num_frames=16  # --- تغییر جدید: اضافه کردن پارامتر تعداد فریم‌ها ---
    ):
        super().__init__(
            gumbel_start_temperature,
            gumbel_end_temperature,
            num_epochs,
            num_frames  # --- تغییر جدید: ارسال به کلاس والد ---
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


def ResNet_50_sparse_uadfv(
    gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200, num_frames=16
):
    return ResNet_sparse(
        block=Bottleneck_sparse,
        num_blocks=[3, 4, 6, 3],
        num_classes=1,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
        dataset_type="uadfv",
        num_frames=num_frames  # --- تغییر جدید: ارسال تعداد فریم‌ها ---
    )
