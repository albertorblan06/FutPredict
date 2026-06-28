"""
Focal Loss for multi-class classification.

Addresses class imbalance in 1X2 prediction by down-weighting easy examples
(surefire favorites) and focusing gradients on hard, ambiguous cases (draws).

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting.
    
    Args:
        alpha: Per-class weight tensor or list. Default [0.25, 0.50, 0.25]
               to up-weight draws (class 1) by 2× relative to wins.
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
               Default 2.0 (standard value from the paper).
        reduction: 'mean', 'sum', or 'none'.
    
    The focal term (1 - p_t)^gamma modulates the standard cross-entropy:
      - When p_t is high (easy example): (1 - 0.95)^2 = 0.0025 → loss ≈ 0
      - When p_t is low (hard example):  (1 - 0.30)^2 = 0.49   → loss preserved
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            alpha = [0.25, 0.50, 0.25]
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C) raw logits from the network
            targets: (N,) integer class labels
        """
        # Move alpha to same device as logits
        if self.alpha.device != logits.device:
            self.alpha = self.alpha.to(logits.device)
        
        # Standard cross-entropy per sample (no reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probability of the true class
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        
        # Alpha weighting per class
        alpha_t = self.alpha.gather(0, targets)
        
        # Combined focal loss
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
