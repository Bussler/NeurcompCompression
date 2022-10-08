import torch
import torch.nn as nn
import numpy as np
from model.SirenLayer import SineLayer, ResidualSineLayer


# M: tries to find the correct amt of features per layer, so that we get target_size neurons
# M: TODO: better way to do this?
def compute_num_neurons(num_layer ,target_size, input_ch=3, output_ch=1):
    d_in = input_ch
    d_out = output_ch

    def network_size(neurons):
        layers = [d_in]
        layers.extend([neurons]*num_layer)
        layers.append(d_out)
        n_layers = len(layers)-1

        n_params = 0
        for ndx in np.arange(n_layers):
            layer_in = layers[ndx]
            layer_out = layers[ndx+1]
            og_layer_in = max(layer_in,layer_out)

            if ndx==0 or ndx==(n_layers-1):
                n_params += ((layer_in+1)*layer_out)
            else:
                is_shortcut = layer_in != layer_out
                if is_shortcut:
                    n_params += (layer_in*layer_out)+layer_out
                n_params += (layer_in*og_layer_in)+og_layer_in
                n_params += (og_layer_in*layer_out)+layer_out
        return n_params

    min_neurons = 16
    while network_size(min_neurons) < target_size:
        min_neurons+=1
    min_neurons-=1

    return min_neurons


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
