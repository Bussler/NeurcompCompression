import torch
import torch.nn as nn
import numpy as np
from model.SirenLayer import SineLayer, ResidualSineLayer


# M: Neurcomp according to https://github.com/matthewberger/neurcomp
class Neurcomp(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, features=[], omega_0=30):
        super(Neurcomp, self).__init__()

        self.d_in = input_ch
        self.d_out = output_ch
        self.omega_0 = omega_0  # M: scale hyperparameter for SIREN

        self.layer_sizes = [self.d_in]
        self.layer_sizes.extend(features)
        self.layer_sizes.append(self.d_out)
        self.n_layers = len(self.layer_sizes) - 1

        self.net_layers = nn.ModuleList()

        for ndx in range(self.n_layers):
            layer_in = self.layer_sizes[ndx]
            layer_out = self.layer_sizes[ndx + 1]
            # print("Debug:", ndx, ndx > 1, ndx == (self.n_layers - 2))
            if ndx != self.n_layers - 1:
                if ndx == 0:
                    # M: first layer
                    self.net_layers.append(SineLayer(layer_in, layer_out, bias=True, is_first=True))
                else:
                    # M: intermed layers
                    self.net_layers.append(ResidualSineLayer(layer_in, bias=True,
                                                             ave_first=ndx > 1,
                                                             ave_second=ndx == (self.n_layers - 2)))
                # M: TODO: append Dropout layers here!
            else:
                final_linear = nn.Linear(layer_in, layer_out)
                with torch.no_grad():
                    # M: uniform init, like in SIREN
                    final_linear.weight.uniform_(-np.sqrt(6 / (layer_in)) / 30.0, np.sqrt(6 / (layer_in)) / 30.0)
                self.net_layers.append(final_linear)
        print("Init of Model done.")

    def forward(self, input):
        out = input
        for ndx, net_layer in enumerate(self.net_layers):
            out = net_layer(out)
        return out
