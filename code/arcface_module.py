"""
ArcFace (Additive Angular Margin) module for deep metric learning.

Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcMarginProduct(nn.Module):
    """
    ArcFace classification head.

    Args:
        in_features: dimension of input features (e.g., 512 for ResNet-18 penultimate).
        out_features: number of classes.
        s: feature scale (default 30.0). Larger s makes logits sharper.
        m: angular margin in radians (default 0.50). Typical range 0.3~0.6.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # threshold: cos(pi - m)
        self.mm = math.sin(math.pi - m) * m  # sin(pi - m) * m

    def forward(self, input, label=None):
        """
        Args:
            input: [B, in_features] normalized features (will be L2-normalized inside).
            label: [B] ground-truth class indices. Required during training.
                   If None, returns cosine logits without angular margin (for eval).
        Returns:
            logits: [B, out_features] scaled by s.
        """
        # L2 normalize both features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # [B, out_features]

        if label is None:
            # Inference mode: no margin, just scaled cosine similarity
            return cosine * self.s

        # Training mode: add angular margin to the ground-truth class
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-6)

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Easy margin: only apply margin when theta < pi - m (i.e., cosine > cos(pi-m))
        # This prevents the margin from pushing too hard on already-hard samples.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode ground-truth labels
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply margin only to ground-truth class, keep others as cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
