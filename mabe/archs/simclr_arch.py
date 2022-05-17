import torch
import torch.nn as nn
import torchvision
from mabe.simclr import SimCLR as simclr


class SimCLR(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt["in_channels"]
        out_emb_size = opt["out_emb_size"]

        encoder = torchvision.models.resnet50(pretrained=True)
        # encoder.load_state_dict(state_dict)
        # Experimental setup for multiplying the grayscale channel
        # https://stackoverflow.com/a/54777347
        weight = encoder.conv1.weight.clone()
        encoder.conv1 = torch.nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # normalize back by in_channels after tiling
        encoder.conv1.weight.data = (
            weight.sum(dim=1, keepdim=True).tile(1, in_channels, 1, 1) / in_channels
        )
        n_features = encoder.fc.in_features
        self.encoder = simclr(encoder, out_emb_size, n_features)
        self.temperature = nn.Parameter(torch.ones(()), requires_grad=True)

    def forward(self, x1, x2, x3):
        h1, h2, h3, z1, z2, z3 = self.encoder(x1, x2, x3)
        # z1 = z1 * self.temperature
        # z2 = z2 * self.temperature
        return h1, h2, h3, z1, z2, z3
