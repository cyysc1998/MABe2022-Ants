# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torchvision.models as models


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, opt):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        dim = opt["dim"]
        K = opt["K"]
        m = opt["m"]
        T = opt["T"]
        mlp = opt["mlp"]
        in_channels = opt["in_channels"]
        base_encoder = models.__dict__["resnet50"]

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        model_url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
        self.encoder_q = self.load_pretrained_weights(self.encoder_q, model_url)
        self.encoder_k = self.load_pretrained_weights(self.encoder_k, model_url)
        self.encoder_q = self.modify_input_for_moco(self.encoder_q, in_channels)
        self.encoder_k = self.modify_input_for_moco(self.encoder_k, in_channels)


        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f"{keys.shape}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def modify_input_for_moco(self, encoder, in_channels):
        weight = encoder.conv1.weight.clone()
        encoder.conv1 = torch.nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # normalize back by in_channels after tiling
        encoder.conv1.weight.data = (
            weight.sum(dim=1, keepdim=True).tile(1, in_channels, 1, 1) / in_channels
        )
        return encoder

    def load_pretrained_weights(self, model, model_url):
        pretrained_dict = torch.utils.model_zoo.load_url(model_url)
        pretrained_dict.pop("fc.weight")
        pretrained_dict.pop("fc.bias")
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def forward(self, im_q, im_k1, im_k2, im_k3, patch, temperature):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)


        # test
        if not self.training:
            return q


        # compute patch features a
        p1_a = self.encoder_k(patch['x1_a'])
        p2_a = self.encoder_k(patch['x1_b'])
        p1_a = nn.functional.normalize(p1_a, dim=1)
        p2_a = nn.functional.normalize(p2_a, dim=1)
        p1_a_gather = concat_all_gather(p1_a)
        p2_a_gather = concat_all_gather(p2_a)
        logits_a1 = p1_a_gather @ p2_a_gather.transpose(1, 0) / temperature
        logits_a2 = p2_a_gather @ p1_a_gather.transpose(1, 0) / temperature


        # compute patch features b
        p1_b = self.encoder_k(patch['x2_a'])
        p2_b = self.encoder_k(patch['x2_b'])
        p1_b = nn.functional.normalize(p1_b, dim=1)
        p2_b = nn.functional.normalize(p2_b, dim=1)
        p1_b_gather = concat_all_gather(p1_b)
        p2_b_gather = concat_all_gather(p2_b)
        logits_b1 = p1_b_gather @ p2_b_gather.transpose(1, 0) / temperature
        logits_b2 = p2_b_gather @ p1_b_gather.transpose(1, 0) / temperature


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # concat all im_k
            im_k = torch.cat((im_k1, im_k2, im_k3), dim=0)

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # split im_k
            k1, k2, k3 = torch.split(k, k.shape[0] // 3, dim=0)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos1 = torch.einsum('nc,nc->n', [q, k1]).unsqueeze(-1)
        l_pos2 = torch.einsum('nc,nc->n', [q, k2]).unsqueeze(-1)
        l_pos3 = torch.einsum('nc,nc->n', [q, k3]).unsqueeze(-1)
        l_pos = torch.cat((l_pos1, l_pos2, l_pos3), dim=0)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_neg = l_neg.repeat(3, 1)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k1)

        return logits, labels, logits_a1, logits_a2, logits_b1, logits_b2


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
