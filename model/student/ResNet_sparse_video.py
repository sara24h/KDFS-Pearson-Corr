import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
# از thop برای محاسبه ساده‌تر FLOPs در مدل‌های استاندارد استفاده می‌کنیم،
# اما برای مدل‌های هرس‌شده (Masked) باید محاسبه را دستی انجام دهیم.
from thop import profile

class SoftMaskedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # این ماسک باید در طول آموزش یاد گرفته شود
        self.mask = nn.Parameter(torch.ones(out_channels, in_channels, *kernel_size))
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, ticket=True):
        if ticket:
            # در حالت آموزش، از ماسک برای هرس کردن وزن‌ها استفاده می‌شود
            masked_weight = self.conv.weight * self.mask
            return F.conv3d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding)
        else:
            # در حالت استنتاج، فقط وزن‌های مهم باقی می‌مانند
            pruned_weight = self.conv.weight * (self.mask > 0.5).float()
            return F.conv3d(x, pruned_weight, self.conv.bias, self.conv.stride, self.conv.padding)

class MaskedNet_3D(nn.Module):
    def __init__(self, gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200):
        super().__init__()
        self.gumbel_start_temperature = gumbel_start_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.num_epochs = num_epochs
        self.gumbel_temperature = gumbel_start_temperature
        self.ticket = False
        self.mask_modules = []

    def update_gumbel_temperature(self, epoch):
        # این تابع برای به‌روزرسانی دمای گامبل در طول آموزش استفاده می‌شود
        self.gumbel_temperature = self.gumbel_start_temperature * math.pow(
            self.gumbel_end_temperature / self.gumbel_start_temperature,
            epoch / self.num_epochs,
        )
        for m in self.mask_modules:
            if hasattr(m, 'update_gumbel_temperature'):
                m.update_gumbel_temperature(self.gumbel_temperature)

    def get_flops(self):
        # این تابع باید برای مدل 3D بازنویسی شود
        device = next(self.parameters()).device
        Flops_total = torch.tensor(0.0, device=device)

        # ابعاد ورودی ویدیویی: (B, C, T, H, W)
        # برای محاسبه FLOPs، فقط به ابعاد T, H, W نیاز داریم
        # مقادیر پیش‌فرض برای یک کلیپ ویدیویی استاندارد
        input_time = getattr(self, "input_time", 16) # طول کلیپ (تعداد فریم‌ها)
        input_size = getattr(self, "input_size", 112) # ارتفاع و عرض فریم‌ها
        
        # محاسبه ابعاد خروجی بعد از لایه اول
        # H_out = floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        conv1_t = (input_time - 7 + 2 * 3) // 2 + 1
        conv1_h = (input_size - 7 + 2 * 3) // 2 + 1
        conv1_w = conv1_h
        
        # FLOPs برای لایه کانولوشن اولیه (3D)
        # فرمول: T_out * H_out * W_out * K_t * K_h * K_w * C_in * C_out
        Flops_total += (
            conv1_t * conv1_h * conv1_w * 7 * 7 * 7 * 3 * 64 # فرض کرنل 7x7x7
        )
        
        # FLOPs برای BatchNorm3d
        Flops_total += conv1_t * conv1_h * conv1_w * 64

        # ابعاد بعد از MaxPool3d
        pool1_t = (conv1_t - 3 + 2 * 1) // 2 + 1
        pool1_h = (conv1_h - 3 + 2 * 1) // 2 + 1
        pool1_w = pool1_h

        # حالا FLOPs را برای بلوک‌های ResNet محاسبه می‌کنیم
        # این بخش باید با توجه به ساختار بلوک‌های 3D شما اصلاح شود
        # در اینجا یک مثال ساده برای یک بلوک با دو کانولوشن 3D ارائه می‌شود
        current_t, current_h, current_w = pool1_t, pool1_h, pool1_w
        for i, m in enumerate(self.mask_modules):
            # فرض می‌کنیم هر m یک SoftMaskedConv3d است
            # تعداد کانال‌های فعال = ماسک‌هایی که فعال هستند
            active_channels = (m.mask.abs() > 0.5).sum().item()
            
            # محاسبه FLOPs برای کانولوشن فعلی
            # C_in باید از لایه قبل گرفته شود
            in_channels = m.in_channels
            kernel_t, kernel_h, kernel_w = m.kernel_size
            
            flops_conv = (
                current_t * current_h * current_w *
                kernel_t * kernel_h * kernel_w *
                in_channels * active_channels
            )
            Flops_total += flops_conv
            
            # به‌روزرسانی ابعاد خروجی برای لایه بعدی
            # این بخش نیاز به دقت دارد، زیرا استراید می‌تواند ابعاد را تغییر دهد
            # در این مثال فرض می‌کنیم استراید 1 است
            # اگر استراید 2 باشد، ابعاد نصف می‌شوند
            # current_t = (current_t + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1
            # ...

        # FLOPs لایه کاملاً متصل (Fully Connected)
        # فرض می‌کنیم ابعاد ویژگی قبل از FC برابر با 512 است
        # این مقدار باید بر اساس ابعاد خروجی آخرین لایه کانولوشنی محاسبه شود
        final_feature_size = 512 * block.expansion # block باید تعریف شود
        Flops_total += final_feature_size * self.fc.out_features

        return Flops_total

# بلوک‌های ResNet-3D
class BasicBlock_3D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = SoftMaskedConv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes),
            )

    def forward(self, x, ticket):
        out = F.relu(self.bn1(self.conv1(x, ticket)))
        out = self.bn2(self.conv2(out, ticket))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class Bottleneck_3D(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = SoftMaskedConv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = SoftMaskedConv3d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes),
            )

    def forward(self, x, ticket):
        out = F.relu(self.bn1(self.conv1(x, ticket)))
        out = F.relu(self.bn2(self.conv2(out, ticket)))
        out = self.bn3(self.conv3(out, ticket))
        out += self.downsample(x)
        out = F.relu(out)
        return out

# مدل ResNet-3D اصلی
class ResNet_3D_sparse(MaskedNet_3D):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=400, # مثلاً برای دیتاست Kinetics-400
        gumbel_start_temperature=2.0,
        gumbel_end_temperature=0.5,
        num_epochs=200,
        input_time=16,    # طول کلیپ ویدیویی
        input_size=112    # اندازه فریم‌ها
    ):
        super().__init__(gumbel_start_temperature, gumbel_end_temperature, num_epochs)
        self.in_planes = 64
        self.input_time = input_time
        self.input_size = input_size

        # لایه‌های 3D
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # ابعاد پس از لایه‌های کانولوشنی باید برای AvgPool محاسبه شود
        # این یک مقدار تقریبی است و باید دقیق محاسبه شود
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) # Adaptive pooling ساده‌تر است
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.mask_modules = [m for m in self.modules() if isinstance(m, SoftMaskedConv3d)]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet_50_sparse_uafdv(num_classes=400):
    return ResNet_3D_sparse(
        block=Bottleneck_3D,
        num_blocks=[3, 4, 6, 3],
        num_classes=num_classes
    )

# --- نحوه استفاده ---
if __name__ == '__main__':
    model = ResNet_50_sparse_uafdv(num_classes=400)
    
    # ایجاد یک ورودی ویدیویی مجازی: (Batch, Channels, Time, Height, Width)
    dummy_input = torch.randn(1, 3, 16, 112, 112)
    
    flops, params = profile(model, inputs=(dummy_input, ))
    print(f"Total FLOPs (with thop): {flops / 1e9:.2f} GFLOPs")
    print(f"Total Params: {params / 1e6:.2f} M")

    custom_flops = model.get_flops()
    print(f"Total FLOPs (custom method): {custom_flops.item() / 1e9:.2f} GFLOPs")
