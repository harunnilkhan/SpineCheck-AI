#simpleunet3plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect # For Dataset check

# ================= Conv Block (3x3) ===================
class ConvBlock(nn.Module):
    """Conv2D + BN + ReLU block (3x3)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# ================= Encoder Block ===================
class EncoderBlock(nn.Module):
    """Encoder: ConvBlock -> ConvBlock -> MaxPool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        e = self.conv1(x)
        e = self.conv2(e)
        p = self.pool(e)
        return e, p

# ================= UNet 3+ (Corrected) ===================
class UNet3Plus(nn.Module):
    """
    Multiscale skip connections with Convolutional blocks for feature fusion.
    Fusion blocks input channels: 5 * base_filters (consistent for all decoders)
    """
    def __init__(self, n_channels=3, n_classes=1, base_filters=64, activation=None):
        super().__init__()
        self.base = base_filters
        self.output_channels = self.base * 5 
        self.n_classes = n_classes
        self.activation = activation

        # Encoder (E1-E5) Channel Configuration
        self.e1_ch = self.base 
        self.e2_ch = self.base * 2
        self.e3_ch = self.base * 4
        self.e4_ch = self.base * 8
        self.e5_ch = self.base * 16

        # --- Encoder ---
        self.e1 = EncoderBlock(n_channels, self.e1_ch)
        self.e2 = EncoderBlock(self.e1_ch, self.e2_ch)
        self.e3 = EncoderBlock(self.e2_ch, self.e3_ch)
        self.e4 = EncoderBlock(self.e3_ch, self.e4_ch)

        # --- Bottleneck (E5) ---
        self.e5_conv1 = ConvBlock(self.e4_ch, self.e5_ch)
        self.e5_conv2 = ConvBlock(self.e5_ch, self.e5_ch)

        # --- Multiscale Skip/Fusion Connection ConvBlocks (Normalize to base_filters) ---
        self.e1_to_f = ConvBlock(self.e1_ch, self.base) 
        self.e2_to_f = ConvBlock(self.e2_ch, self.base) 
        self.e3_to_f = ConvBlock(self.e3_ch, self.base) 
        self.e4_to_f = ConvBlock(self.e4_ch, self.base) 
        self.e5_to_f = ConvBlock(self.e5_ch, self.base) 
        self.d_to_f = ConvBlock(self.output_channels, self.base)

        # --- Decoder Fusion Conv Blocks (Output size is consistent: self.output_channels) ---
        self.conv_d4 = ConvBlock(self.output_channels, self.output_channels)
        self.conv_d3 = ConvBlock(self.output_channels, self.output_channels)
        self.conv_d2 = ConvBlock(self.output_channels, self.output_channels)
        self.conv_d1 = ConvBlock(self.output_channels, self.output_channels)

        # --- Final Output ---
        self.outconv = nn.Conv2d(self.output_channels, n_classes, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self):
        # Kaiming Normal initialization for Conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _resize_like(self, x, target, mode='bilinear'):
        """
        Resize tensor x to match the spatial size of target, handling align_corners for non-nearest modes.
        """
        if mode in ('nearest', 'area', 'nearest-exact'):
            # Do not use align_corners for these modes
            return F.interpolate(x, size=target.shape[2:], mode=mode)
        else:
            # Use align_corners=True for bilinear and similar modes
            return F.interpolate(x, size=target.shape[2:], mode=mode, align_corners=True)

    def forward(self, x):
        # Encoder Path
        e1, p1 = self.e1(x) 
        e2, p2 = self.e2(p1)
        e3, p3 = self.e3(p2)
        e4, p4 = self.e4(p3)
        e5 = self.e5_conv1(p4)
        e5 = self.e5_conv2(e5)

        # --- Decoder D4 Fusion (E1, E2, E3, E4, E5) ---
        # E1 to D4: MaxPool 8x
        t_e1 = F.max_pool2d(e1, 8); t_e1 = self.e1_to_f(t_e1)
        # E2 to D4: MaxPool 4x
        t_e2 = F.max_pool2d(e2, 4); t_e2 = self.e2_to_f(t_e2)
        # E3 to D4: MaxPool 2x
        t_e3 = F.max_pool2d(e3, 2); t_e3 = self.e3_to_f(t_e3)
        # E4 to D4: No scaling
        t_e4 = self.e4_to_f(e4)
        # E5 to D4: UpSample 2x (Bilinear)
        t_e5 = self._resize_like(e5, e4); t_e5 = self.e5_to_f(t_e5)
        d4 = self.conv_d4(torch.cat([t_e1, t_e2, t_e3, t_e4, t_e5], dim=1))

        # --- Decoder D3 Fusion (E1, E2, E3, D4, E5) ---
        # E1 to D3: MaxPool 4x
        t_e1 = F.max_pool2d(e1, 4); t_e1 = self.e1_to_f(t_e1)
        # E2 to D3: MaxPool 2x
        t_e2 = F.max_pool2d(e2, 2); t_e2 = self.e2_to_f(t_e2)
        # E3 to D3: No scaling
        t_e3 = self.e3_to_f(e3)
        # D4 to D3: UpSample 2x (Bilinear)
        t_d4 = self._resize_like(d4, e3); t_d4 = self.d_to_f(t_d4) 
        # E5 to D3: UpSample 4x (Nearest)
        t_e5 = self._resize_like(e5, e3, mode='nearest'); t_e5 = self.e5_to_f(t_e5)
        d3 = self.conv_d3(torch.cat([t_e1, t_e2, t_e3, t_d4, t_e5], dim=1))

        # --- Decoder D2 Fusion (E1, E2, D3, D4, E5) ---
        # E1 to D2: MaxPool 2x
        t_e1 = F.max_pool2d(e1, 2); t_e1 = self.e1_to_f(t_e1)
        # E2 to D2: No scaling
        t_e2 = self.e2_to_f(e2)
        # D3 to D2: UpSample 2x (Bilinear)
        t_d3 = self._resize_like(d3, e2); t_d3 = self.d_to_f(t_d3) 
        # D4 to D2: UpSample 4x (Nearest)
        t_d4 = self._resize_like(d4, e2, mode='nearest'); t_d4 = self.d_to_f(t_d4)
        # E5 to D2: UpSample 8x (Nearest)
        t_e5 = self._resize_like(e5, e2, mode='nearest'); t_e5 = self.e5_to_f(t_e5)
        d2 = self.conv_d2(torch.cat([t_e1, t_e2, t_d3, t_d4, t_e5], dim=1))

        # --- Decoder D1 Fusion (E1, D2, D3, D4, E5) ---
        # E1 to D1: No scaling
        t_e1 = self.e1_to_f(e1)
        # D2 to D1: UpSample 2x (Bilinear)
        t_d2 = self._resize_like(d2, e1); t_d2 = self.d_to_f(t_d2) 
        # D3 to D1: UpSample 4x (Nearest)
        t_d3 = self._resize_like(d3, e1, mode='nearest'); t_d3 = self.d_to_f(t_d3)
        # D4 to D1: UpSample 8x (Nearest)
        t_d4 = self._resize_like(d4, e1, mode='nearest'); t_d4 = self.d_to_f(t_d4)
        # E5 to D1: UpSample 16x (Nearest)
        t_e5 = self._resize_like(e5, e1, mode='nearest'); t_e5 = self.e5_to_f(t_e5)
        d1 = self.conv_d1(torch.cat([t_e1, t_d2, t_d3, t_d4, t_e5], dim=1))

        # Output
        y = self.outconv(d1)
        if self.activation == 'sigmoid':
            y = torch.sigmoid(y)
        elif self.activation == 'softmax':
            y = torch.softmax(y, dim=1)

        return y