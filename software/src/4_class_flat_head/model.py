from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedAudioBackbone(nn.Module):
    """
    Lightweight FP32 2D CNN using strictly FINN-compatible operations:
      (B, 1, 128, 128) -> (B, 64) acoustic embedding.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> (16, 64, 64)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> (32, 32, 32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # -> (64, 16, 16)

        # Fixed kernel/stride AvgPool2d (FINN-legal replacement for AdaptiveAvgPool2d)
        self.global_pool = nn.AvgPool2d(kernel_size=16, stride=16)  # -> (64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.pool2(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = self.pool3(F.relu(self.bn3(self.conv3(x)), inplace=True))
        x = self.global_pool(x)
        return torch.flatten(x, 1)  # (B, 64)


class FourClassICBHIModel(nn.Module):
    """
    Single shared backbone + controlled demographic fusion feeding a direct
    4-way classification head: [Healthy (0), COPD (1), Pneumonia (2), URTI (3)].
    """

    def __init__(self, demo_dim: int = 6, demo_dropout_p: float = 0.50):
        super().__init__()
        self.demo_dropout_p = demo_dropout_p
        self.backbone = SharedAudioBackbone()

        self.demo_mlp = nn.Sequential(
            nn.Linear(demo_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
        )

        # Shared multimodal fusion layer: 64 audio + 16 demo = 80 -> 64
        self.fusion = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )

        # Direct 4-class diagnosis head
        self.classifier_head = nn.Linear(64, 4)

    def _apply_modality_dropout(self, demographics: torch.Tensor) -> torch.Tensor:
        if self.training and self.demo_dropout_p > 0.0:
            keep_prob = 1.0 - self.demo_dropout_p
            mask = torch.bernoulli(
                torch.full((demographics.size(0), 1), keep_prob, device=demographics.device)
            )
            return demographics * mask
        return demographics

    def _forward_single_channel(
        self, spectrogram: torch.Tensor, demographics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        audio_feat = self.backbone(spectrogram)
        demo_in = self._apply_modality_dropout(demographics)
        demo_feat = self.demo_mlp(demo_in)

        shared = self.fusion(torch.cat([audio_feat, demo_feat], dim=-1))
        logits = self.classifier_head(shared)
        log_probs = F.log_softmax(logits, dim=-1)

        return {
            "logits": logits,
            "log_probs": log_probs,
            "tree_log_probs": log_probs,  # Kept so predict_subset() in train.py works unchanged
        }

    def forward(
        self,
        spectrogram: torch.Tensor,
        demographics: torch.Tensor,
        return_per_channel: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Accepts:
          - 4D spectrogram (B, 1, 128, 128) for single-stethoscope training/eval
          - 5D spectrogram (B, 8, 1, 128, 128) for 8-channel acoustic array inference,
            aggregating across channels via log-probability averaging.
        """
        if spectrogram.dim() == 5:
            b, ch, c, f, t = spectrogram.shape
            flat_spec = spectrogram.view(b * ch, c, f, t)
            demo_rep = demographics.unsqueeze(1).expand(b, ch, -1).reshape(b * ch, -1)

            out = self._forward_single_channel(flat_spec, demo_rep)
            ch_log_probs = out["log_probs"].view(b, ch, 4)
            agg_log_probs = F.log_softmax(ch_log_probs.mean(dim=1), dim=-1)
            agg_logits = out["logits"].view(b, ch, 4).mean(dim=1)

            res = {
                "logits": agg_logits,
                "log_probs": agg_log_probs,
                "tree_log_probs": agg_log_probs,
            }
            if return_per_channel:
                res["per_channel_probs"] = torch.exp(ch_log_probs)
            return res

        return self._forward_single_channel(spectrogram, demographics)


# Alias so existing imports in train.py continue to work without modification
HierarchicalICBHIModel = FourClassICBHIModel