import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class Bottleneck(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2d(x)


class GlobalUpConv2d(nn.Module):

    def __init__(self, in_ch, h_ch, out_ch):
        super().__init__()
        self.lat = nn.Conv2d(in_ch, h_ch, 1)
        self.top = nn.Conv2d(h_ch, out_ch, 3, padding=1)

    def forward(self, xs):
        x1 = xs[0]
        x2 = xs.pop()
        x2 = self.lat(x2)
        x2 = F.interpolate(x1, scale_factor=2) + x2
        x2 = self.top(x2)
        xs.insert(0, x2)
        return xs


class GlobalNet(nn.Module):
    """
    GlobalNet part of CascadedPyramidNetwork
    it take a list of features, from low level to high level, convert it to a list of features
    in the same order using a combining strategy introduced in CPN paper
    it first apply a lateral conv to each feature to generate feature of same depth(maybe means they got same semantic)
    then it generate current level feature by up sample higher one and add with current one, then apply 3x3 conv to it
    for highest feature, only lateral is applied, for lowest onw last 3x3 conv will used to generate the output
    """

    def __init__(self, in_chs, h_ch, out_ch):
        """
        :param in_chs:
            list of int, number of channels of each level of features, from low level to high level
        :param h_ch:
            int, number of hidden channels
        :param out_ch:
            int, number of channels of lowest level feature
        """

        super().__init__()
        up_layers = [GlobalUpConv2d(in_ch, h_ch, h_ch) for in_ch in in_chs[1:-1]]
        up_layers.insert(0, GlobalUpConv2d(in_chs[0], h_ch, out_ch))
        self.up = nn.Sequential(*reversed(up_layers))
        self.lat = nn.Conv2d(in_chs[-1], h_ch, 1)

    def forward(self, xs):
        xs.insert(0, self.lat(xs.pop()))
        xs = self.up(xs)
        return xs


class RefineUpConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, level):
        super().__init__()
        scale = 1 << level
        if scale == 1:
            self.up = lambda x: x
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2 * scale, stride=scale, padding=scale // 2)

    def forward(self, xs):
        x = xs.pop()
        x = self.up(x)
        xs.insert(0, x)
        return xs


class RefineNet(nn.Module):
    """
    RefineNet part of CascadedPyramidNetwork
    """

    def __init__(self, in_chs, h_ch):
        super().__init__()
        up_layers = [RefineUpConv2d(in_ch, h_ch, i) for i, in_ch in enumerate(in_chs)]
        self.up = nn.Sequential(*reversed(up_layers))

    def forward(self, x):
        x = self.up(x)
        return x


class BodyNet(nn.Module):
    """
    feature extractor
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class CascadePyramidNet(nn.Module):

    def __init__(self, num_attrs):
        super().__init__()

        self.body_net = BodyNet()

        in_chs = [256, 512, 1024, 2048]
        self.global_net = GlobalNet(in_chs, 256, num_attrs)

        in_chs = [256 for _ in range(len(in_chs))]
        in_chs[0] = num_attrs
        self.refine_net = RefineNet(in_chs, 256)

        self.out = nn.Conv2d(sum(in_chs), num_attrs, 1)

    def forward(self, x):
        x = self.body_net(x)
        x = self.global_net(x)
        x = self.refine_net(x)
        return self.out(torch.cat(x, dim=1))
