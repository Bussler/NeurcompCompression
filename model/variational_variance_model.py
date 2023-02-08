import torch
import torch.nn as nn
import torch.nn.functional as F


class Variance_Model(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, n_layers=4, size_layers=32):
        super(Variance_Model, self).__init__()

        self.net_layers = nn.ModuleList(
            [nn.Linear(input_ch, size_layers)] +
            [nn.Linear(size_layers, size_layers) for i in range(n_layers - 1)]
        )

        self.final_layer = nn.Linear(size_layers, output_ch)


    def forward(self, input):
        out = input
        for ndx, net_layer in enumerate(self.net_layers):
            out = F.relu(net_layer(out))
        out = self.final_layer(out)
        return out
