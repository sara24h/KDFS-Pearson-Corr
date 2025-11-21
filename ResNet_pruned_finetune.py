import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    """Bottleneck Block برای ResNet-50 با پشتیبانی از pruning"""
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, masks=None):
        super(BottleneckBlock, self).__init__()
        
        self.masks = masks if masks is not None else [None, None, None]
        
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # تعداد کانال‌های فعال برای conv3
        if self.masks[2] is not None:
            num_active_channels = (self.masks[2] == 1).sum().item()
            self.conv3 = nn.Conv2d(out_channels, num_active_channels, 
                                   kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        else:
            self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                                   kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        """Forward pass با پشتیبانی از masked pruning"""
        identity = x
        
        # Conv1 + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Conv2 + BN + ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Conv3 + BN (با پشتیبانی از mask)
        if self.masks[2] is not None:
            # محاسبه feature map از conv3
            feature_map = self.conv3(out)
            batch_size, _, h, w = feature_map.shape
            
            # ایجاد padded feature map با ابعاد کامل
            padded_feature_map = torch.zeros(
                batch_size, 
                len(self.masks[2]), 
                h, 
                w,
                device=feature_map.device,
                dtype=feature_map.dtype
            )
            
            # کپی کردن feature_map به کانال‌های فعال
            active_channels = self.masks[2] == 1
            padded_feature_map[:, active_channels, :, :] = feature_map
            
            out = self.bn3(padded_feature_map)
        else:
            out = self.conv3(out)
            out = self.bn3(out)
        
        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet_50_pruned_hardfakevsreal(nn.Module):
    """ResNet-50 pruned model برای دسته‌بندی fake vs real"""
    
    def __init__(self, masks=None, num_classes=1):
        super(ResNet_50_pruned_hardfakevsreal, self).__init__()
        
        self.masks = masks if masks is not None else [None] * 16
        self.in_channels = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 3, stride=1, masks_start_idx=0)
        self.layer2 = self._make_layer(128, 4, stride=2, masks_start_idx=3)
        self.layer3 = self._make_layer(256, 6, stride=2, masks_start_idx=7)
        self.layer4 = self._make_layer(512, 3, stride=2, masks_start_idx=13)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride, masks_start_idx):
        """ساخت یک layer از bottleneck blocks"""
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * BottleneckBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BottleneckBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleneckBlock.expansion)
            )
        
        layers = []
        
        # اولین block
        block_masks = self.masks[masks_start_idx:masks_start_idx + 3] if self.masks else None
        layers.append(BottleneckBlock(
            self.in_channels, 
            out_channels, 
            stride, 
            downsample,
            masks=block_masks
        ))
        
        self.in_channels = out_channels * BottleneckBlock.expansion
        
        # بقیه blocks
        for i in range(1, num_blocks):
            idx = masks_start_idx + i * 3
            block_masks = self.masks[idx:idx + 3] if self.masks else None
            layers.append(BottleneckBlock(
                self.in_channels,
                out_channels,
                masks=block_masks
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights for the model"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        output = self.fc(features)
        
        return output, features


model = ResNet_50_pruned_hardfakevsreal(masks=your_masks)
