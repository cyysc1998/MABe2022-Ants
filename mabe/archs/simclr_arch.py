import torch
import torch.nn as nn
import torchvision


class SimCLR(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt["in_channels"]
        out_emb_size = opt["out_emb_size"]

        self.encoder = torchvision.models.resnet50(pretrained=False)
        state_dict = torch.load("/cache/resnet50-0676ba61.pth")
        self.encoder.load_state_dict(state_dict)
        # Experimental setup for multiplying the grayscale channel
        # https://stackoverflow.com/a/54777347
        weight = self.encoder.conv1.weight.clone()
        self.encoder.conv1 = torch.nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # normalize back by in_channels after tiling
        self.encoder.conv1.weight.data = (
            weight.sum(dim=1, keepdim=True).tile(1, in_channels, 1, 1) / in_channels
        )
        # Replace the fc layer with an MLP projector
        n_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, out_emb_size, bias=False),
        )
        # temperature
        self.temperature = nn.Parameter(torch.ones(()), requires_grad=True)

    def forward(self, x_list):
        return [self.encoder(x) for x in x_list]
