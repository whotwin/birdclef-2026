"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss for SimCLR.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning.

    For a batch of N pairs (each audio has 2 views),
    computes similarity between all 2N representations.

    Args:
        temperature: scaling factor for cosine similarity (default: 0.07)
        device: torch device
    """

    def __init__(self, temperature=0.07, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z1, z2):
        """
        Args:
            z1: [N, D] normalized projections for view 1
            z2: [N, D] normalized projections for view 2
        Returns:
            scalar loss
        """
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2N, D]

        # Cosine similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]

        # Mask out self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=self.device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(N, 2 * N, device=self.device),
            torch.arange(0, N, device=self.device)
        ])  # [2N]

        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels)
        return loss


def infonce_loss(z1, z2, temperature=0.07):
    """
    Functional version of NT-Xent loss.

    Args:
        z1, z2: [N, D] each, already normalized
        temperature: float
    Returns:
        scalar loss
    """
    return NTXentLoss(temperature=temperature)(z1, z2)
