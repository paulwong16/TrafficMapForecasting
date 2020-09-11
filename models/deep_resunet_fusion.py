import torch
from torch import nn
import torch.nn.functional as F
from models.deep_resunet import DeepResUNet


class CNN(nn.Module):
    def __init__(self, in_channels=1, out=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, out, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class Deep_Res_UNet_fusion(nn.Module):
    def __init__(self, in_channels=1, out_classes=2, static_in=1, static_out=2):
        super(Deep_Res_UNet_fusion, self).__init__()
        self.unet = DeepResUNet(in_channels, out_classes)
        self.cnn = CNN(static_in, static_out)
        self.conv = nn.Conv2d(out_classes+static_out, out_classes, 3, 1, 1)
        self.in_ch = in_channels
        self.out_ch = out_classes
        self.s_in_ch = static_in
        self.s_out_ch = static_out

    def forward(self, x):
        x1 = self.unet(x[:, :self.in_ch, :, :])
        x2 = self.cnn(x[:, self.in_ch:, :, :])
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.conv(x))
        return x
