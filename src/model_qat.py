"""
Drop-in replacement for `software/src/model_qat.py`
Repository: https://github.com/sbiruduganti/IEEE-EOH-Acoustic-Array-Lung-Monitor

Contains:
1. `SharedAudioBackboneQAT`:
   - Accepts `(Batch, 1, 128, 128)` spectrograms from `dataset.py`.
   - Adds `self.input_quant` (`Uint8ActPerTensorFloat`) so Xilinx FINN's `Streamline` pass succeeds.
   - Replaces `AdaptiveAvgPool2d((4, 4))` with `nn.MaxPool2d(kernel_size=4, stride=4)` on `(64, 16, 16)`
     producing the exact `(Batch, 1024)` output vector without breaking FINN's HLS compiler.
2. `ICBHIModelQAT`:
   - Defaults to `num_classes=2` (binary specialist gate) while keeping the exact
     `demo_mlp` (`5 -> 16 -> 32`) and `1056 -> 128 -> num_classes` fusion head.
3. `BinaryDiagnosisCascadeQAT`:
   - Wraps the 3 binary specialist gates (`gate1`: Healthy vs. Diseased, `gate2`: COPD vs. Acute,
     `gate3`: Pneumonia vs. URTI) and outputs calibrated 4-class probabilities `[Healthy, COPD, Pneumonia, URTI]`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import brevitas.nn as qnn
    from brevitas.quant import (
        Int8ActPerTensorFloat,
        Uint8ActPerTensorFloat,
        Int8WeightPerTensorFloat,
        Int8Bias,
    )
    BREVITAS_AVAILABLE = True
except ImportError:
    BREVITAS_AVAILABLE = False


class SharedAudioBackboneQAT(nn.Module):
    def __init__(self, weight_bit_width: int = 4, act_bit_width: int = 4):
        super().__init__()
        stem_bits = max(8, weight_bit_width)

        if BREVITAS_AVAILABLE:
            self.input_quant = qnn.QuantIdentity(
                act_quant=Uint8ActPerTensorFloat, bit_width=8, return_quant_tensor=False
            )
            # (1, 128, 128) -> (16, 64, 64) [8-bit stem preserves subtle fine crackles]
            self.conv1 = qnn.QuantConv2d(
                1, 16, kernel_size=3, padding=1, bias=False,
                weight_quant=Int8WeightPerTensorFloat, weight_bit_width=stem_bits,
            )
            self.bn1 = nn.BatchNorm2d(16)
            self.relu1 = qnn.QuantReLU(
                act_quant=Uint8ActPerTensorFloat, bit_width=stem_bits, return_quant_tensor=False
            )
            self.pool1 = nn.MaxPool2d(2, 2)

            # (16, 64, 64) -> (32, 32, 32)
            self.conv2 = qnn.QuantConv2d(
                16, 32, kernel_size=3, padding=1, bias=False,
                weight_quant=Int8WeightPerTensorFloat, weight_bit_width=weight_bit_width,
            )
            self.bn2 = nn.BatchNorm2d(32)
            self.relu2 = qnn.QuantReLU(
                act_quant=Uint8ActPerTensorFloat, bit_width=act_bit_width, return_quant_tensor=False
            )
            self.pool2 = nn.MaxPool2d(2, 2)

            # (32, 32, 32) -> (64, 16, 16) -> MaxPool2d(4, 4) -> (64, 4, 4) = 1024
            self.conv3 = qnn.QuantConv2d(
                32, 64, kernel_size=3, padding=1, bias=False,
                weight_quant=Int8WeightPerTensorFloat, weight_bit_width=weight_bit_width,
            )
            self.bn3 = nn.BatchNorm2d(64)
            self.relu3 = qnn.QuantReLU(
                act_quant=Uint8ActPerTensorFloat, bit_width=act_bit_width, return_quant_tensor=False
            )
            self.pool3 = nn.MaxPool2d(2, 2)
            self.pool_final = nn.MaxPool2d(kernel_size=4, stride=4)
        else:
            self.input_quant = nn.Identity()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu1 = nn.ReLU(inplace=True)
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(32)
            self.relu2 = nn.ReLU(inplace=True)
            self.pool2 = nn.MaxPool2d(2, 2)

            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(64)
            self.relu3 = nn.ReLU(inplace=True)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.pool_final = nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool_final(x)
        return torch.flatten(x, 1)  # (Batch, 1024)


class ICBHIModelQAT(nn.Module):
    def __init__(self, num_classes: int = 2, weight_bit_width: int = 4, act_bit_width: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = SharedAudioBackboneQAT(
            weight_bit_width=weight_bit_width, act_bit_width=act_bit_width
        )

        if BREVITAS_AVAILABLE:
            self.demo_mlp = nn.Sequential(
                qnn.QuantLinear(
                    5, 16, bias=True,
                    weight_quant=Int8WeightPerTensorFloat, weight_bit_width=8, bias_quant=Int8Bias
                ),
                qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat, bit_width=8, return_quant_tensor=False),
                qnn.QuantLinear(
                    16, 32, bias=True,
                    weight_quant=Int8WeightPerTensorFloat, weight_bit_width=8, bias_quant=Int8Bias
                ),
                qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat, bit_width=8, return_quant_tensor=False),
            )
            self.classifier = nn.Sequential(
                qnn.QuantLinear(
                    1056, 128, bias=True,
                    weight_quant=Int8WeightPerTensorFloat, weight_bit_width=weight_bit_width, bias_quant=Int8Bias
                ),
                qnn.QuantReLU(act_quant=Uint8ActPerTensorFloat, bit_width=act_bit_width, return_quant_tensor=False),
                nn.Dropout(p=0.3),
                qnn.QuantLinear(
                    128, num_classes, bias=True,
                    weight_quant=Int8WeightPerTensorFloat, weight_bit_width=8, bias_quant=Int8Bias
                ),
            )
        else:
            self.demo_mlp = nn.Sequential(
                nn.Linear(5, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 32),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Linear(1056, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_classes),
            )

    def forward(
        self,
        spectrogram: torch.Tensor,
        demographics: torch.Tensor,
        return_per_channel: bool = False,
    ):
        if spectrogram.dim() == 5:
            b, ch, c, f, t = spectrogram.shape
            flat_spec = spectrogram.view(b * ch, c, f, t)
            ch_feats = self.backbone(flat_spec).view(b, ch, 1024)

            if return_per_channel:
                demo_feat = self.demo_mlp(demographics).unsqueeze(1).expand(b, ch, 32)
                fused_ch = torch.cat([ch_feats, demo_feat], dim=-1).view(b * ch, 1056)
                ch_logits = self.classifier(fused_ch).view(b, ch, self.num_classes)
                pooled_feats = ch_feats.max(dim=1)[0]
                pooled_logits = self.classifier(torch.cat([pooled_feats, self.demo_mlp(demographics)], dim=-1))
                return pooled_logits, ch_logits

            audio_feat = ch_feats.max(dim=1)[0]
        else:
            audio_feat = self.backbone(spectrogram)

        demo_feat = self.demo_mlp(demographics)
        fused = torch.cat([audio_feat, demo_feat], dim=-1)
        return self.classifier(fused)


class BinaryDiagnosisCascadeQAT(nn.Module):
    """
    Combines the 3 binary specialist gates (`gate1`, `gate2`, `gate3`) into a single
    callable module that returns 4-class probabilities `[Healthy, COPD, Pneumonia, URTI]`.
    """

    def __init__(self, weight_bit_width: int = 4, act_bit_width: int = 4):
        super().__init__()
        self.gate1 = ICBHIModelQAT(num_classes=2, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.gate2 = ICBHIModelQAT(num_classes=2, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.gate3 = ICBHIModelQAT(num_classes=2, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

    @staticmethod
    def _combine_tree_probs(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        p_healthy = p1[..., 0:1]
        p_diseased = p1[..., 1:2]
        p_copd = p_diseased * p2[..., 0:1]
        p_acute = p_diseased * p2[..., 1:2]
        p_pneumonia = p_acute * p3[..., 0:1]
        p_urti = p_acute * p3[..., 1:2]
        return torch.cat([p_healthy, p_copd, p_pneumonia, p_urti], dim=-1)

    def forward(
        self,
        spectrogram: torch.Tensor,
        demographics: torch.Tensor,
        return_per_channel: bool = False,
    ):
        if return_per_channel and spectrogram.dim() == 5:
            g1_pool, g1_ch = self.gate1(spectrogram, demographics, return_per_channel=True)
            g2_pool, g2_ch = self.gate2(spectrogram, demographics, return_per_channel=True)
            g3_pool, g3_ch = self.gate3(spectrogram, demographics, return_per_channel=True)

            pool_probs = self._combine_tree_probs(
                F.softmax(g1_pool, dim=-1), F.softmax(g2_pool, dim=-1), F.softmax(g3_pool, dim=-1)
            )
            ch_probs = self._combine_tree_probs(
                F.softmax(g1_ch, dim=-1), F.softmax(g2_ch, dim=-1), F.softmax(g3_ch, dim=-1)
            )
            return pool_probs, ch_probs

        p1 = F.softmax(self.gate1(spectrogram, demographics), dim=-1)
        p2 = F.softmax(self.gate2(spectrogram, demographics), dim=-1)
        p3 = F.softmax(self.gate3(spectrogram, demographics), dim=-1)
        return self._combine_tree_probs(p1, p2, p3)