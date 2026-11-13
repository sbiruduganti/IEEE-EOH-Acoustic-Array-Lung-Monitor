import torch
import torch.nn as nn
import brevitas.nn as qnn


class QuantBasicBlock2d(nn.Module):
    def __init__(self, in_planes, planes, stride=1, bit_width=8):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_bit_width=bit_width
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False)

        self.conv2 = qnn.QuantConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=bit_width
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                qnn.QuantConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False, weight_bit_width=bit_width),
                nn.BatchNorm2d(planes),
                qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False),
            )
        self.relu_out = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu_out(out + self.shortcut(x))


class QuantResNet18LungModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, bit_width=8):
        super().__init__()
        self.quant_spec = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False)
        # Immediate 2x2 MaxPool in stem cuts spatial grid from 64x64 -> 32x32 (4x CPU speedup!)
        self.stem = nn.Sequential(
            qnn.QuantConv2d(in_channels, 24, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=bit_width),
            nn.BatchNorm2d(24),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            nn.MaxPool2d(2, 2),
        )
        self.layer1 = QuantBasicBlock2d(24, 32, stride=1, bit_width=bit_width)   # 32x32
        self.layer2 = QuantBasicBlock2d(32, 64, stride=2, bit_width=bit_width)   # 16x16
        self.layer3 = QuantBasicBlock2d(64, 96, stride=2, bit_width=bit_width)   # 8x8
        self.layer4 = QuantBasicBlock2d(96, 128, stride=2, bit_width=bit_width)  # 4x4
        self.pool = nn.MaxPool2d(kernel_size=(4, 4))  # Static [B, 128, 1, 1] for Xilinx FINN

        self.quant_demo = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False)
        self.demo_mlp = nn.Sequential(
            qnn.QuantLinear(5, 16, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            qnn.QuantLinear(16, 32, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            qnn.QuantLinear(128 + 32, 64, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            qnn.QuantLinear(64, num_classes, bias=True, weight_bit_width=bit_width),
        )

    def forward(self, spectrogram, demographics):
        x = self.quant_spec(spectrogram)
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        audio_feat = torch.flatten(self.pool(x), 1)
        demo_feat = self.demo_mlp(self.quant_demo(demographics))
        return self.classifier(torch.cat([audio_feat, demo_feat], dim=1))


class QuantDualStreamLungModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, bit_width=8):
        super().__init__()
        # Stream A: 2D-CNN on [B, 3, 64, 64] (Log-Mel + Delta + Delta-Delta)
        self.quant_spec = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False)
        self.cnn2d = nn.Sequential(
            qnn.QuantConv2d(in_channels, 24, kernel_size=3, padding=1, bias=False, weight_bit_width=bit_width),
            nn.BatchNorm2d(24),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            nn.MaxPool2d(2, 2),  # 32x32

            qnn.QuantConv2d(24, 48, kernel_size=3, padding=1, bias=False, weight_bit_width=bit_width),
            nn.BatchNorm2d(48),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            nn.MaxPool2d(2, 2),  # 16x16

            qnn.QuantConv2d(48, 96, kernel_size=3, padding=1, bias=False, weight_bit_width=bit_width),
            nn.BatchNorm2d(96),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            nn.MaxPool2d(2, 2),  # 8x8

            qnn.QuantConv2d(96, 128, kernel_size=3, padding=1, bias=False, weight_bit_width=bit_width),
            nn.BatchNorm2d(128),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            nn.MaxPool2d(kernel_size=(8, 8)),  # [B, 128, 1, 1]
        )

        # Stream B: Fast 1D-CNN on [B, 2, 256] Transient Crackle Envelope
        self.quant_wave = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False)
        self.cnn1d = nn.Sequential(
            qnn.QuantConv1d(2, 24, kernel_size=5, padding=2, bias=False, weight_bit_width=bit_width),
            nn.BatchNorm1d(24),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            nn.MaxPool1d(4),  # 256 -> 64

            qnn.QuantConv1d(24, 48, kernel_size=5, padding=2, bias=False, weight_bit_width=bit_width),
            nn.BatchNorm1d(48),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            nn.MaxPool1d(4),  # 64 -> 16

            qnn.QuantConv1d(48, 64, kernel_size=3, padding=1, bias=False, weight_bit_width=bit_width),
            nn.BatchNorm1d(64),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            nn.MaxPool1d(16),  # 16 -> 1 -> [B, 64]
        )

        # Stream C: Quantized Demographic + Cycle Length + Chest Location MLP
        self.quant_demo = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=False)
        self.demo_mlp = nn.Sequential(
            qnn.QuantLinear(5, 16, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            qnn.QuantLinear(16, 32, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            qnn.QuantLinear(128 + 64 + 32, 64, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
            qnn.QuantLinear(64, num_classes, bias=True, weight_bit_width=bit_width),
        )

    def forward(self, wave_env, spectrogram, demographics):
        feat_2d = torch.flatten(self.cnn2d(self.quant_spec(spectrogram)), 1)  # [B, 128]
        feat_1d = torch.flatten(self.cnn1d(self.quant_wave(wave_env)), 1)     # [B, 64]
        feat_demo = self.demo_mlp(self.quant_demo(demographics))              # [B, 32]
        return self.classifier(torch.cat([feat_2d, feat_1d, feat_demo], dim=1))