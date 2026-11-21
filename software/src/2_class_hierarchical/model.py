from typing import Dict, Tuple
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


class HierarchicalICBHIModel(nn.Module):
    """
    Single shared backbone + controlled demographic fusion feeding the 3-gate
    hierarchical decision tree:
      - Gate 1: Healthy (0) vs. Diseased (1)
      - Gate 2: COPD (0) vs. Acute Infection (1)
      - Gate 3: Pneumonia (0) vs. URTI (1)

    Outputs exact 4-class tree log-probabilities [Healthy, COPD, Pneumonia, URTI]
    alongside individual gate logits for conditional auxiliary supervision.
    """

    def __init__(self, demo_dim: int = 6, demo_dropout_p: float = 0.40):
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

        # Three 2-class hierarchical gate heads sharing the same backbone
        self.gate1_head = nn.Linear(64, 2)  # Healthy vs. Diseased
        self.gate2_head = nn.Linear(64, 2)  # COPD vs. Acute
        self.gate3_head = nn.Linear(64, 2)  # Pneumonia vs. URTI

    def _apply_modality_dropout(self, demographics: torch.Tensor) -> torch.Tensor:
        if self.training and self.demo_dropout_p > 0.0:
            keep_prob = 1.0 - self.demo_dropout_p
            mask = torch.bernoulli(
                torch.full((demographics.size(0), 1), keep_prob, device=demographics.device)
            )
            return demographics * mask
        return demographics

    @staticmethod
    def _gate_logits_to_tree_log_probs(
        g1_logits: torch.Tensor, g2_logits: torch.Tensor, g3_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes exact 4-class log-probabilities down the binary probability tree:
          log P(Healthy)   = log P_g1(0)
          log P(COPD)      = log P_g1(1) + log P_g2(0)
          log P(Pneumonia) = log P_g1(1) + log P_g2(1) + log P_g3(0)
          log P(URTI)      = log P_g1(1) + log P_g2(1) + log P_g3(1)
        """
        lp1 = F.log_softmax(g1_logits, dim=-1)
        lp2 = F.log_softmax(g2_logits, dim=-1)
        lp3 = F.log_softmax(g3_logits, dim=-1)

        lp_healthy = lp1[..., 0:1]
        lp_diseased = lp1[..., 1:2]
        lp_copd = lp_diseased + lp2[..., 0:1]
        lp_acute = lp_diseased + lp2[..., 1:2]
        lp_pneumonia = lp_acute + lp3[..., 0:1]
        lp_urti = lp_acute + lp3[..., 1:2]

        return torch.cat([lp_healthy, lp_copd, lp_pneumonia, lp_urti], dim=-1)

    def _forward_single_channel(
        self, spectrogram: torch.Tensor, demographics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        audio_feat = self.backbone(spectrogram)
        demo_in = self._apply_modality_dropout(demographics)
        demo_feat = self.demo_mlp(demo_in)

        shared = self.fusion(torch.cat([audio_feat, demo_feat], dim=-1))
        g1_logits = self.gate1_head(shared)
        g2_logits = self.gate2_head(shared)
        g3_logits = self.gate3_head(shared)
        tree_log_probs = self._gate_logits_to_tree_log_probs(g1_logits, g2_logits, g3_logits)

        return {
            "tree_log_probs": tree_log_probs,
            "g1_logits": g1_logits,
            "g2_logits": g2_logits,
            "g3_logits": g3_logits,
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
            ch_log_probs = out["tree_log_probs"].view(b, ch, 4)
            # Average log-probabilities across the 8 microphones, then re-normalize
            agg_log_probs = F.log_softmax(ch_log_probs.mean(dim=1), dim=-1)

            res = {
                "tree_log_probs": agg_log_probs,
                "g1_logits": out["g1_logits"].view(b, ch, 2).mean(dim=1),
                "g2_logits": out["g2_logits"].view(b, ch, 2).mean(dim=1),
                "g3_logits": out["g3_logits"].view(b, ch, 2).mean(dim=1),
            }
            if return_per_channel:
                res["per_channel_probs"] = torch.exp(ch_log_probs)
            return res

        return self._forward_single_channel(spectrogram, demographics)