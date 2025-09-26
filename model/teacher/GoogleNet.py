
import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_planes, n1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes, n3x3red, kernel_size=1),
            BasicConv2d(n3x3red, n3x3, kernel_size=3, padding=1)
        )
        
        # <<< این نسخه صحیح بر اساس آخرین خطا است >>>
        self.branch3 = nn.Sequential(
            BasicConv2d(in_planes, n5x5red, kernel_size=1),
            BasicConv2d(n5x5red, n5x5, kernel_size=3, padding=1) # <-- فقط دو لایه
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_planes, pool_planes, kernel_size=1)
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat([y1, y2, y3, y4], 1)

class GoogLeNet(nn.Module):
    def __init__(self, block=Inception, num_classes=1):
        super(GoogLeNet, self).__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.inception3a = block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)
        self.inception4a = block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)
        self.inception5a = block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # تعریف لیست خالی برای ذخیره ویژگی‌ها
        feature_list = []

        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)
        out = self.inception3a(out)
        out = self.inception3b(out)
        # اضافه کردن اولین گروه از ویژگی‌ها به لیست
        feature_list.append(out)

        out = self.maxpool3(out)
        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        # اضافه کردن دومین گروه از ویژگی‌ها به لیست
        feature_list.append(out)

        out = self.maxpool4(out)
        out = self.inception5a(out)
        out = self.inception5b(out)
        # اضافه کردن سومین گروه از ویژگی‌ها به لیست
        feature_list.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        
        # حالا این خط بدون خطا اجرا می‌شود
        return out, feature_list

def GoogLeNet_deepfake():
    return GoogLeNet(block=Inception, num_classes=1)
