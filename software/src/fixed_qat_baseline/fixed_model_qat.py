import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn

class SharedAudioBackboneQAT(nn.Module):
    def __init__(self, input_bit_width=8, weight_bit_width=4, act_bit_width=4):
        super(SharedAudioBackboneQAT, self).__init__()
        
        # FIX PROBLEM 2: Explicit Quant identity at input node for FPGA streams
        self.quant_inp = qnn.QuantIdentity(bit_width=input_bit_width, return_quant_tensor=False)

        # Conv Layer 1
        self.conv1 = qnn.QuantConv2d(1, 16, kernel_size=3, padding=1, weight_bit_width=weight_bit_width)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width)
        
        # Conv Layer 2
        self.conv2 = qnn.QuantConv2d(16, 32, kernel_size=3, padding=1, weight_bit_width=weight_bit_width)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width)

        # Conv Layer 3
        self.conv3 = qnn.QuantConv2d(32, 64, kernel_size=3, padding=1, weight_bit_width=weight_bit_width)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = qnn.QuantReLU(bit_width=act_bit_width)
        
        # Conv Layer 4: Additional feature expansion layer
        self.conv4 = qnn.QuantConv2d(64, 64, kernel_size=3, padding=1, weight_bit_width=weight_bit_width)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = qnn.QuantReLU(bit_width=act_bit_width)
        
        # FIX PROBLEM 1: Replaced AdaptiveAvgPool2d with standard fixed 2x2 MaxPool2d layers
        # Input size: 128x128 -> Pool1: 64x64 -> Pool2: 32x32 -> Pool3: 16x16 -> Pool4: 8x8 -> Pool5: 4x4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.quant_inp(x)                         # [B, 1, 128, 128]
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))  # [B, 16, 64, 64]
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))  # [B, 32, 32, 32]
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))  # [B, 64, 16, 16]
        x = self.pool(self.relu4(self.bn4(self.conv4(x))))  # [B, 64, 8, 8]
        x = self.pool(x)                                    # [B, 64, 4, 4] -> 1024 features
        return torch.flatten(x, 1)

class ICBHIModelQAT(nn.Module):
    def __init__(self, num_classes=4, input_bit_width=8, weight_bit_width=4, act_bit_width=4):
        super(ICBHIModelQAT, self).__init__()
        self.backbone = SharedAudioBackboneQAT(
            input_bit_width=input_bit_width,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width
        )
        
        self.demo_mlp = nn.Sequential(
            qnn.QuantLinear(5, 16, weight_bit_width=weight_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            qnn.QuantLinear(16, 32, weight_bit_width=weight_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width)
        )

        self.classifier = nn.Sequential(
            qnn.QuantLinear(1024 + 32, 128, weight_bit_width=weight_bit_width),
            qnn.QuantReLU(bit_width=act_bit_width),
            nn.Dropout(p=0.3),
            qnn.QuantLinear(128, num_classes, weight_bit_width=weight_bit_width)
        )

    def forward(self, spectrogram, demographics, return_per_channel=False):
        if spectrogram.ndim == 5 or (spectrogram.ndim == 4 and spectrogram.shape[1] == 8):
            if spectrogram.ndim == 5:
                batch_size, num_mics, C, F_dim, T_dim = spectrogram.shape
                x = spectrogram.view(batch_size * num_mics, C, F_dim, T_dim)
            else:
                batch_size, num_mics, F_dim, T_dim = spectrogram.shape
                x = spectrogram.view(batch_size * num_mics, 1, F_dim, T_dim)
            
            channel_feats = self.backbone(x)
            demo_feat = self.demo_mlp(demographics)

            if return_per_channel:
                demo_feat_expanded = demo_feat.repeat_interleave(num_mics, dim=0)
                fused_per_ch = torch.cat((channel_feats, demo_feat_expanded), dim=1)
                logits_per_ch = self.classifier(fused_per_ch)
                return logits_per_ch.view(batch_size, num_mics, 4)

            channel_feats = channel_feats.view(batch_size, num_mics, 1024)
            audio_feat, _ = torch.max(channel_feats, dim=1)
            fused_features = torch.cat((audio_feat, demo_feat), dim=1)
            return self.classifier(fused_features)
        else:
            audio_feat = self.backbone(spectrogram)
            demo_feat = self.demo_mlp(demographics)
            fused_features = torch.cat((audio_feat, demo_feat), dim=1)
            return self.classifier(fused_features)