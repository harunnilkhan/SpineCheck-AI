#attentionunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ Conv Block ============
class ConvBlock(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2"""
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

# ============ Attention Gate ============
class AttentionBlock(nn.Module):
    """Applies channel-wise attention to the skip connection (x) using the gate signal (g)"""
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        # Linear transformations for gate and skip connection
        self.W_g = nn.Conv2d(g_channels, inter_channels, 1, bias=True)
        self.W_x = nn.Conv2d(x_channels, inter_channels, 1, bias=True)
        # Activation for output
        self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # Resize g to match x's spatial dimensions implicitly handled by UNet structure
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Calculate attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        
        # Apply attention to the skip connection
        return x * psi

# ============ Attention U-Net ============
class AttentionUNet(nn.Module):
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

        # Decoder + Attention Blocks
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.att4 = AttentionBlock(base_filters * 8, base_filters * 8, base_filters * 4)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8)

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.att3 = AttentionBlock(base_filters * 4, base_filters * 4, base_filters * 2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.att2 = AttentionBlock(base_filters * 2, base_filters * 2, base_filters)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.att1 = AttentionBlock(base_filters, base_filters, base_filters // 2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)

        self.outconv = nn.Conv2d(base_filters, n_classes, 1)

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder Path with Attention Gates
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        # Output
        y = self.outconv(d1)
        if self.activation == 'sigmoid':
            y = torch.sigmoid(y)
        elif self.activation == 'softmax':
            y = torch.softmax(y, dim=1)
        return y