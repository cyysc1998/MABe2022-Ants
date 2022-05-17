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


def info_nce_loss(embed, seq_id):
    x1, x2, pos_x = embed
    x1 = torch.cat(GatherLayer.apply(x1), dim=0)
    x2 = torch.cat(GatherLayer.apply(x2), dim=0)
    pos_x = torch.cat(GatherLayer.apply(pos_x), dim=0)
    x = torch.cat([x1, x2, pos_x], dim=0)
    
    n_views = len(embed)
    seq_id = torch.cat(GatherLayer.apply(seq_id), dim=0)
    seq_id = torch.cat([seq_id for _ in range(n_views)], dim=0)
    loss = contrastive_loss(x, seq_id, n_views)
    
    return loss


def get_mask(sequence_id, n_views):
    b = len(sequence_id)
    mask_id = sequence_id[None, :] - sequence_id[:, None]
    zero_id = torch.where(mask_id == 0, 1, 0)
    one_indices = torch.cat([torch.arange(b // n_views) for i in range(n_views)], dim=0)
    one_indices = (one_indices.unsqueeze(0) == one_indices.unsqueeze(1)).long()
    mask = zero_id + one_indices
    pos_mask = torch.where(mask > 0, 1, 0)
    pos_mask.fill_diagonal_(0)
    neg_mask = torch.where(mask > 0, 0, 1)
    return pos_mask, neg_mask


def contrastive_loss(x, sequence_id, n_views, temperature=1):
    B, _ = x.shape
    logits = nn.CosineSimilarity(dim=2)(x.unsqueeze(1), x.unsqueeze(0))
    pos_mask, neg_mask = get_mask(sequence_id, n_views)
    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits + neg_logits)
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    loss = -mean_log_prob_pos * temperature
    loss = loss.mean()

    return loss
