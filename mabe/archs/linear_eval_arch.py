import torch
import torch.nn as nn


class LinearEval(nn.Module):
    def __init__(self, opt):
        super().__init__()
        input_dim = opt["input_dim"]
        output_dim = opt["output_dim"]
        self.fc = nn.Linear(input_dim, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)
