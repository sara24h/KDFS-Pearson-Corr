import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from .layer import SoftMaskedConv2d


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
            if hasattr(m, 'checkpoint'):
                m.load_state_dict(m.checkpoint)

    def update_gumbel_temperature(self, epoch):
        self.gumbel_temperature = self.gumbel_start_temperature * math.pow(
            self.gumbel_end_temperature / self.gumbel_start_temperature,
            epoch / self.num_epochs,
        )
        for m in self.mask_modules:
            m.update_gumbel_temperature(self.gumbel_temperature)

    def get_flops(self):
        """
        محاسبه FLOPs با در نظر گرفتن ماسک‌های Gumbel-Softmax
        """
        device = next(self.parameters()).device
        Flops_total = torch.tensor(0.0, device=device)

        # دیکشنری اندازه تصاویر ورودی برای هر دیتاست
        image_sizes = {
            "hardfakevsrealfaces": 300,
            "rvf10k": 256,
            "140k": 256,
            "uadfv": 256
        }

        dataset_type = getattr(self, "dataset_type", "hardfakevsrealfaces")
        input_size = image_sizes.get(dataset_type, 256)

        # محاسبه FLOPs برای لایه‌های اولیه (conv1, bn1)
        conv1_h = (input_size - 7 + 2 * 3) // 2 + 1  # example
        maxpool_h = (conv1_h - 3 + 2 * 1) // 2 + 1   # example

        Flops_total = Flops_total + (
            conv1_h * conv1_h * 7 * 7 * 3 * 64 +  # FLOPs for conv1
            conv1_h * conv1_h * 64                 # FLOPs for bn1
        )

        # محاسبه FLOPs برای بلوک‌های ResNet با کانولوشن‌های ماسک‌دار
        for i, m in enumerate(self.mask_modules):
            m = m.to(device)

            # محاسبه تعداد فیلترهای فعال با استفاده از Gumbel-Softmax
            with torch.no_grad():
                if self.ticket:
                    # حالت inference: از ماسک باینری استفاده کن
                    mask_probs = (torch.sigmoid(m.mask_weight) > 0.5).float()
                else:
                    # حالت training: از Gumbel-Softmax استفاده کن
                    mask_probs = F.gumbel_softmax(
                        logits=m.mask_weight,
                        tau=m.gumbel_temperature,
                        hard=False,
                        dim=1
                    )[:, 1, :, :]  # احتمال فعال بودن (کلاس 1)

                # میانگین احتمال فعال بودن فیلترها
                active_filters = mask_probs.mean().item() * m.out_channels

            Flops_shortcut_conv = 0
            Flops_shortcut_bn = 0

            # محاسبات مخصوص ResNet-50 (48 لایه ماسک‌دار: 16 بلوک × 3 لایه)
            if len(self.mask_modules) == 48:
                # در هر بلوک Bottleneck سه لایه conv داریم:
                # - conv1: 1x1, in_channels -> planes
                # - conv2: 3x3, planes -> planes
                # - conv3: 1x1, planes -> planes*4

                block_idx = i // 3  # شماره بلوک
                layer_in_block = i % 3  # لایه در بلوک (0, 1, 2)

                if layer_in_block == 0:  # conv1 در Bottleneck
                    # لایه اول: ورودی از بلوک قبل (یا ورودی اصلی)
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w *
                        m.kernel_size * m.kernel_size *
                        m.in_channels * active_filters
                    )

                elif layer_in_block == 1:  # conv2 در Bottleneck
                    # لایه دوم: ورودی از لایه قبل همین بلوک
                    if i > 0:
                        prev_m = self.mask_modules[i - 1]
                        with torch.no_grad():
                            if self.ticket:
                                prev_mask_probs = (torch.sigmoid(prev_m.mask_weight) > 0.5).float()
                            else:
                                prev_mask_probs = F.gumbel_softmax(
                                    logits=prev_m.mask_weight,
                                    tau=prev_m.gumbel_temperature,
                                    hard=False,
                                    dim=1
                                )[:, 1, :, :]
                            prev_active_filters = prev_mask_probs.mean().item() * prev_m.out_channels
                    else:
                        prev_active_filters = m.in_channels

                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w *
                        m.kernel_size * m.kernel_size *
                        prev_active_filters * active_filters
                    )

                else:  # layer_in_block == 2, conv3 در Bottleneck
                    # لایه سوم: ورودی از لایه قبل همین بلوک
                    if i > 0:
                        prev_m = self.mask_modules[i - 1]
                        with torch.no_grad():
                            if self.ticket:
                                prev_mask_probs = (torch.sigmoid(prev_m.mask_weight) > 0.5).float()
                            else:
                                prev_mask_probs = F.gumbel_softmax(
                                    logits=prev_m.mask_weight,
                                    tau=prev_m.gumbel_temperature,
                                    hard=False,
                                    dim=1
                                )[:, 1, :, :]
                            prev_active_filters = prev_mask_probs.mean().item() * prev_m.out_channels
                    else:
                        prev_active_filters = m.in_channels

                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w *
                        m.kernel_size * m.kernel_size *
                        prev_active_filters * active_filters
                    )

                    # محاسبه FLOPs برای downsample (اتصال کوتاه) در صورت نیاز
                    # downsample فقط در اولین بلوک هر stage اتفاق می‌افتد
                    # stage boundaries: block 0, 3, 7, 13 (برای [3,4,6,3])
                    stage_first_blocks = [0, 3, 7, 13]
                    if block_idx in stage_first_blocks and getattr(m, "stride", 1) != 1:
                        # downsample: 1x1 conv از in_channels به out_channels
                        Flops_shortcut_conv = (
                            m.feature_map_h * m.feature_map_w *
                            1 * 1 * m.in_channels * m.out_channels
                        )
                        Flops_shortcut_bn = m.feature_map_h * m.feature_map_w * m.out_channels

                # FLOPs برای BatchNorm
                Flops_bn = m.feature_map_h * m.feature_map_w * active_filters

            else:
                # برای معماری‌های دیگر (مثل ResNet-18/34)
                Flops_conv = (
                    m.feature_map_h * m.feature_map_w *
                    m.kernel_size * m.kernel_size *
                    m.in_channels * active_filters
                )
                Flops_bn = m.feature_map_h * m.feature_map_w * active_filters

            Flops_total = (
                Flops_total + Flops_conv + Flops_bn +
                Flops_shortcut_conv + Flops_shortcut_bn
            )

        # اضافه کردن FLOPs برای لایه‌های پایانی (avgpool و fc)
        # avgpool: تقریباً بدون FLOPs
        # fc: 512 * expansion * num_classes
        expansion = 4  # برای Bottleneck
        Flops_total = Flops_total + (512 * expansion * self.fc.out_features)

        return Flops_total

    def get_video_flops(self, video_duration_seconds=None, fps=None):
        """
        محاسبه FLOPs برای پردازش یک ویدیو کامل
        """
        flops_per_frame = self.get_flops()
        num_sampled_frames = 16  # تعداد فریم‌هایی که از ویدیو نمونه‌برداری می‌شود
        total_video_flops = flops_per_frame * num_sampled_frames
        return total_video_flops

    def get_retention_rate(self):
        """
        محاسبه درصد فیلترهای باقی‌مانده (retention rate)
        """
        if len(self.mask_modules) == 0:
            return 1.0

        total_active = 0
        total_filters = 0

        with torch.no_grad():
            for m in self.mask_modules:
                if isinstance(m, SoftMaskedConv2d):
                    if self.ticket:
                        mask_probs = (torch.sigmoid(m.mask_weight) > 0.5).float()[:, 1, 0, 0]
                    else:
                        mask_probs = F.gumbel_softmax(
                            logits=m.mask_weight,
                            tau=m.gumbel_temperature,
                            hard=False,
                            dim=1
                        )[:, 1, 0, 0]

                    total_active += mask_probs.sum().item()
                    total_filters += m.out_channels

        retention_rate = total_active / total_filters if total_filters > 0 else 1.0
        return retention_rate


class BasicBlock_sparse(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SoftMaskedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
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
        self.conv2 = SoftMaskedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SoftMaskedConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
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
    def __init__(self, block, num_blocks, num_classes=1, gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200, dataset_type="hardfakevsrealfaces"):
        super().__init__(gumbel_start_temperature, gumbel_end_temperature, num_epochs)
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
        self.feat1 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=1)
        self.feat2 = nn.Conv2d(128 * block.expansion, 128 * block.expansion, kernel_size=1)
        self.feat3 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=1)
        self.feat4 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=1)
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


def ResNet_50_sparse_uadfv(gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200):
    return ResNet_sparse(block=Bottleneck_sparse, num_blocks=[3, 4, 6, 3], num_classes=1, gumbel_start_temperature=gumbel_start_temperature, gumbel_end_temperature=gumbel_end_temperature, num_epochs=num_epochs, dataset_type="uadfv")


def ResNet_50_sparse_rvf10k(gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200):
    return ResNet_sparse(block=Bottleneck_sparse, num_blocks=[3, 4, 6, 3], num_classes=1, gumbel_start_temperature=gumbel_start_temperature, gumbel_end_temperature=gumbel_end_temperature, num_epochs=num_epochs, dataset_type="rvf10k")
