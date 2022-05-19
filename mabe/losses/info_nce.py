import torch
import torch.distributed as dist
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


def info_nce_loss(x1, x2):
    x1 = torch.cat(GatherLayer.apply(x1), dim=0)
    x2 = torch.cat(GatherLayer.apply(x2), dim=0)

    B = x1.shape[0]
    x = torch.cat([x1, x2])

    logits = x @ x.T
    mask = torch.eye(2 * B).to(logits)
    logits = logits.masked_fill(mask == 1, float("-inf"))

    labels = (torch.Tensor(list(range(2 * B))) + B) % (2 * B)
    labels = labels.to(logits).long()

    loss = F.cross_entropy(logits, labels)

    return loss
