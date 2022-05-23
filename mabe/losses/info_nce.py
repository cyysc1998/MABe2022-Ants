import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.

    From: https://github.com/spijkervet/SimCLR/blob/HEAD/simclr/modules/gather.py
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def info_nce_loss(p1, p2, z1, z2):
    p1 = torch.cat(GatherLayer.apply(p1), dim=0)
    p2 = torch.cat(GatherLayer.apply(p2), dim=0)
    z1 = torch.cat(GatherLayer.apply(z1), dim=0)
    z2 = torch.cat(GatherLayer.apply(z2), dim=0)

    criterion = nn.CosineSimilarity(dim=1)
    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

    return loss
