'''
Training utilities for dermoscopic melanoma classification.
'''
import torch
import torch.nn as nn
import numpy as np
import random

def entropy_weighted_loss(logits, targets):
    """
    Computes a weighted cross-entropy loss based on the entropy of the predicted probabilities.
    Args:
        logits (torch.Tensor): Raw output logits from the model of shape [B, C] where B is batch size and C is number of classes.
        targets (torch.Tensor): Ground truth labels of shape [B] with class indices.
        Returns:
        torch.Tensor: The computed loss value.
    """
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)

    # Entropy = -sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=1)  # [B]
    entropy = entropy / (torch.max(entropy) + 1e-8)  # normalize: [0,1]

    base_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
    weighted_loss = entropy * base_loss
    return weighted_loss.mean()


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    Args:
        seed (int): The seed value to set for random number generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
