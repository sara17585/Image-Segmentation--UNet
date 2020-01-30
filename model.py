import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.first = downStep(1, 64, first=True)
        self.down1 = downStep(64, 128)
        self.down2 = downStep(128, 256)
        self.down3 = downStep(256, 512)
        self.down4 = downStep(512, 1024)
        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64, withrelu= False)
        self.last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # todo
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.last(x)

        return x


class downStep(nn.Module):
    def __init__(self, inC, outC, first=False):
        super(downStep, self).__init__()
        # todo
        self.first = first
        self.max = nn.MaxPool2d(2, 2)
        self.conv = TwoConv(inC, outC)

    def forward(self, x):
        # todo
        if  self.first== False:
            x = self.max(x)
        x = self.conv(x)

        return x


class upStep(nn.Module):
    def __init__(self, inC, outC, withrelu=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.up = nn.ConvTranspose2d(inC, outC, 2, stride=2)
        self.withrelu = withrelu
        self.conv = TwoConv(inC, outC,self.withrelu)

    def crop(self, layer, target_size):
        diff_y = (layer.shape[2]- target_size[0]) // 2
        diff_x = (layer.shape[3] - target_size[1]) // 2
        temp = layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
        return temp

    def forward(self, x, bridge):
        # todo
        x = self.up(x)
        crop1 = self.crop(bridge, x.shape[2:])
        x = torch.cat([x, crop1], dim=1)
        x = self.conv(x)
        return x


class TwoConv(nn.Module):
    def __init__(self, inC, outC, withrelu=True):
        super(TwoConv, self).__init__()
        self.withrelu = withrelu
        self.conv = nn.Conv2d(inC, outC, 3)
        self.norm = nn.BatchNorm2d(outC)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outC, outC, 3)

    def forward(self, x):
        if self.withrelu:
            x = self.conv(x)
            x = self.relu(x)
            x = self.norm(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.norm(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.conv2(x)
            x = self.norm(x)

        return x

