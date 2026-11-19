import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss dynamically focuses training on hard/ambiguous respiratory samples
    and down-weights easy majority-class predictions (Normal).
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights tensor
        self.gamma = gamma  # Focusing parameter
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        
        # Apply label smoothing to targets
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Compute focal factor: (1 - p_t)^gamma
        focal_factor = torch.pow(1.0 - probs, self.gamma)
        
        loss = -focal_factor * smooth_targets * log_probs

        if self.alpha is not None:
            alpha_weight = self.alpha.unsqueeze(0)
            loss = loss * alpha_weight

        return loss.sum(dim=1).mean()