#unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ Conv Block ============
class ConvBlock(nn.Module):
    """Conv2D + BN + ReLU x2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# ============ U-Net ============
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_filters=64, activation=None):
        super().__init__()
        self.activation = activation

        # Encoder
        self.enc1 = ConvBlock(n_channels, base_filters)
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8) # Combined channels: 8*B + 8*B = 16*B

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4) # Combined channels: 4*B + 4*B = 8*B

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2) # Combined channels: 2*B + 2*B = 4*B

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters) # Combined channels: B + B = 2*B

        # Output
        self.outconv = nn.Conv2d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder Path
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Output
        y = self.outconv(d1)
        if self.activation == 'sigmoid':
            y = torch.sigmoid(y)
        elif self.activation == 'softmax':
            y = torch.softmax(y, dim=1)
        return y