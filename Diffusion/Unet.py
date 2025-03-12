import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, T=1000, ch=128, ch_mult=[1,2,3,4], attn=[2], num_res_blocks=2, dropout=0.1):
        super(UNet, self).__init__()

        # Encoder
        self.inc = DoubleConv(3, ch)
        self.down1 = Down(ch, ch * ch_mult[0])
        self.down2 = Down(ch * ch_mult[0], ch * ch_mult[1])
        self.down3 = Down(ch * ch_mult[1], ch * ch_mult[2])
        self.down4 = Down(ch * ch_mult[2], ch * ch_mult[3])

        # Decoder (corrected skip connection channels)
        self.up1 = Up(ch * ch_mult[3], ch * ch_mult[2], ch * ch_mult[2])
        self.up2 = Up(ch * ch_mult[2], ch * ch_mult[1], ch * ch_mult[1])
        self.up3 = Up(ch * ch_mult[1], ch * ch_mult[0], ch * ch_mult[0])
        self.up4 = Up(ch * ch_mult[0], ch, ch)

        self.outc = nn.Conv2d(ch, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)
