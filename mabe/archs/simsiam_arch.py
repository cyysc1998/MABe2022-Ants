import torch
import torch.nn as nn
import torchvision


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, opt):
        super(SimSiam, self).__init__()
        dim = opt["pred_dim"]
        pred_dim = opt["out_emb_size"]
        in_channels = opt["in_channels"]
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = torchvision.models.resnet50(pretrained=True, num_classes=dim, zero_init_residual=True)
        weight = self.encoder.conv1.weight.clone()
        self.encoder.conv1 = torch.nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # normalize back by in_channels after tiling
        self.encoder.conv1.weight.data = (
            weight.sum(dim=1, keepdim=True).tile(1, in_channels, 1, 1) / in_channels
        )

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.temperature = nn.Parameter(torch.ones(()), requires_grad=True)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC
        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()