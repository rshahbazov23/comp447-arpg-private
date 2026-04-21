"""
Per-token confidence scoring for the ARPG rejection mechanism.

Each function takes logits of shape (batch, num_pred, vocab_size) and returns
a confidence tensor of shape (batch, num_pred) where HIGHER means MORE confident.

These operate on the SAME logits used for sampling (post temperature, post top-k/p
filtering) so the confidence distribution matches what was actually sampled from.
"""

import torch
import torch.nn.functional as F


def max_prob(logits: torch.Tensor) -> torch.Tensor:
    """Maximum softmax probability per position. Range: [0, 1]."""
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Negated predictive entropy per position. Higher = more confident.

    H(p) = -sum(p_i * log(p_i)). We return -H so the sign convention
    matches the other metrics (higher = more confident).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # -H(p) = sum(p * log p), already has the correct sign
    return (probs * log_probs).sum(dim=-1)


def margin(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 softmax probability per position. Range: [0, 1]."""
    probs = F.softmax(logits, dim=-1)
    top2 = probs.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


CONFIDENCE_FNS = {
    "max_prob": max_prob,
    "entropy": entropy,
    "margin": margin,
}
