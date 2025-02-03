import torch.nn as nn
import torch.nn.functional as F
import torch

######################################################
####################### UNET #########################
######################################################

class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2,2),
            TripleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SpecialDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AdaptiveMaxPool2d((8,8)),
            TripleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = TripleConv(in_channels + in_channels // 2, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SpecialUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(size=(13,13), mode='bilinear', align_corners=False)
        self.conv = TripleConv(in_channels + in_channels // 2, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        #Encoder
        self.inc = TripleConv(3, 64)
        self.down1 = SpecialDown(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        #MLP
        self.fc1 = nn.Linear(2 * 2 * 512, 2048)
        self.fc2 = nn.Linear(2048,2 * 2 * 512)

        #Decoder
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = SpecialUp(128, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        #Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = torch.flatten(x4, 1)

        #MLP conjunto
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        #Decoder
        x = x.view(x.size(0), 512, 2, 2)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
    