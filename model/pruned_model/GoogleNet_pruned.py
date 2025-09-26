import torch
import torch.nn as nn


def get_preserved_filter_num(mask):
    return int(mask.sum())


class Inception_pruned(nn.Module):
    def __init__(
        self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, masks
    ):
        super().__init__()
        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes

        # 1x1 conv branch
        if self.n1x1:
            self.branch1x1 = nn.Sequential(
                nn.Conv2d(in_planes, get_preserved_filter_num(masks[0]), kernel_size=1),
                nn.BatchNorm2d(get_preserved_filter_num(masks[0])),
                nn.ReLU(True),
            )

        # 1x1 conv -> 3x3 conv branch
        if self.n3x3:
            self.branch3x3 = nn.Sequential(
                nn.Conv2d(in_planes, get_preserved_filter_num(masks[1]), kernel_size=1),
                nn.BatchNorm2d(get_preserved_filter_num(masks[1])),
                nn.ReLU(True),
                nn.Conv2d(
                    get_preserved_filter_num(masks[1]),
                    get_preserved_filter_num(masks[2]),
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(get_preserved_filter_num(masks[2])),
                nn.ReLU(True),
            )

        # 1x1 conv -> 5x5 conv branch
        if self.n5x5 > 0:
            self.branch5x5 = nn.Sequential(
                nn.Conv2d(in_planes, get_preserved_filter_num(masks[3]), kernel_size=1),
                nn.BatchNorm2d(get_preserved_filter_num(masks[3])),
                nn.ReLU(True),
                nn.Conv2d(
                    get_preserved_filter_num(masks[3]),
                    get_preserved_filter_num(masks[4]),
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(get_preserved_filter_num(masks[4])),
                nn.ReLU(True),
                nn.Conv2d(
                    get_preserved_filter_num(masks[4]),
                    get_preserved_filter_num(masks[5]),
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(get_preserved_filter_num(masks[5])),
                nn.ReLU(True),
            )

        # 3x3 pool -> 1x1 conv branch
        if self.pool_planes > 0:
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_planes, get_preserved_filter_num(masks[6]), kernel_size=1),
                nn.BatchNorm2d(get_preserved_filter_num(masks[6])),
                nn.ReLU(True),
            )

    def forward(self, x):
        out = []
        if self.n1x1:
            y1 = self.branch1x1(x)
            out.append(y1)
        if self.n3x3:
            y2 = self.branch3x3(x)
            out.append(y2)
        if self.n5x5:
            y3 = self.branch5x5(x)
            out.append(y3)
        if self.pool_planes:
            y4 = self.branch_pool(x)
            out.append(y4)
        return torch.cat(out, 1)


class GoogLeNet_pruned(nn.Module):
    def __init__(self, block=Inception_pruned, filters=None, num_classes=1, masks=[]):
        super().__init__()

        assert len(masks) == 63

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        if filters is None:
            filters = [
                [64, 128, 32, 32], [128, 192, 96, 64], [192, 208, 48, 64],
                [160, 224, 64, 64], [128, 256, 64, 64], [112, 288, 64, 64],
                [256, 320, 128, 128], [256, 320, 128, 128], [384, 384, 128, 128],
            ]

        # RENAMED layers to match teacher model (e.g., inception3a)
        self.inception3a = block(64, filters[0][0], 96, filters[0][1], 16, filters[0][2], filters[0][3], masks=masks[0:7])
        self.inception3b = block(
            sum([get_preserved_filter_num(mask) for mask in [masks[0], masks[2], masks[5], masks[6]]]),
            filters[1][0], 128, filters[1][1], 32, filters[1][2], filters[1][3], masks=masks[7:14]
        )

        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = block(
            sum([get_preserved_filter_num(mask) for mask in [masks[7], masks[9], masks[12], masks[13]]]),
            filters[2][0], 96, filters[2][1], 16, filters[2][2], filters[2][3], masks=masks[14:21]
        )
        self.inception4b = block(
            sum([get_preserved_filter_num(mask) for mask in [masks[14], masks[16], masks[19], masks[20]]]),
            filters[3][0], 112, filters[3][1], 24, filters[3][2], filters[3][3], masks=masks[21:28]
        )
        self.inception4c = block(
            sum([get_preserved_filter_num(mask) for mask in [masks[21], masks[23], masks[26], masks[27]]]),
            filters[4][0], 128, filters[4][1], 24, filters[4][2], filters[4][3], masks=masks[28:35]
        )
        self.inception4d = block(
            sum([get_preserved_filter_num(mask) for mask in [masks[28], masks[30], masks[33], masks[34]]]),
            filters[5][0], 144, filters[5][1], 32, filters[5][2], filters[5][3], masks=masks[35:42]
        )
        self.inception4e = block(
            sum([get_preserved_filter_num(mask) for mask in [masks[35], masks[37], masks[40], masks[41]]]),
            filters[6][0], 160, filters[6][1], 32, filters[6][2], filters[6][3], masks=masks[42:49]
        )

        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = block(
            sum([get_preserved_filter_num(mask) for mask in [masks[42], masks[44], masks[47], masks[48]]]),
            filters[7][0], 160, filters[7][1], 32, filters[7][2], filters[7][3], masks=masks[49:56]
        )
        self.inception5b = block(
            sum([get_preserved_filter_num(mask) for mask in [masks[49], masks[51], masks[54], masks[55]]]),
            filters[8][0], 192, filters[8][1], 48, filters[8][2], filters[8][3], masks=masks[56:63]
        )

        # FIXED: Changed from AvgPool2d(8) to AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # RENAMED: from linear to fc
        self.fc = nn.Linear(
            sum([get_preserved_filter_num(mask) for mask in [masks[56], masks[58], masks[61], masks[62]]]),
            num_classes,
        )

    def forward(self, x):
        feature_list = []
        out = self.pre_layers(x)
        
        # RENAMED layers in forward pass
        out = self.inception3a(out)
        out = self.inception3b(out)
        feature_list.append(out)

        out = self.maxpool1(out)
        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        feature_list.append(out)

        out = self.maxpool2(out)
        out = self.inception5a(out)
        out = self.inception5b(out)
        feature_list.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # RENAMED: from linear to fc
        out = self.fc(out)
        return out, feature_list



def GoogLeNet_pruned_deepfake(masks):
    return GoogLeNet_pruned(block=Inception_pruned, num_classes=1, masks=masks)
