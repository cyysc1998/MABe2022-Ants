import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(logits, labels, inter_split):
    ce = nn.CrossEntropyLoss(reduction='none')
    loss = ce(logits, labels)
    intra_loss = loss[:inter_split].mean()
    inter_loss = loss[inter_split:].mean()
    return intra_loss, inter_loss
