import torch
import torch.nn as nn
import numpy as np
from model.SirenLayer import SineLayer, ResidualSineBlock
from model.SmallifyDropoutLayer import SmallifyDropout, SmallifyResidualSiren
from model.VariationalDropoutLayer import VariationalDropout


# M: Neurcomp according to "Compressive Neural Representations of Volumetric Scalar Fields"
class Neurcomp(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, features=[], omega_0=30, dropout_technique='',
                 sign_variance_momentum=0.02, use_resnet=True, pruning_threshold=0.9, variational_init_droprate=0.5):
        super(Neurcomp, self).__init__()

        self.d_in = input_ch
        self.d_out = output_ch
        self.omega_0 = omega_0  # M: scale hyperparameter for SIREN
        self.use_resnet = use_resnet

        self.layer_sizes = [self.d_in]
        self.layer_sizes.extend(features)
        self.layer_sizes.append(self.d_out)
        self.n_layers = len(self.layer_sizes) - 1

        self.net_layers = nn.ModuleList()

        # M: setup dropout
        dropout_layer = None
        if dropout_technique and '_quant' not in dropout_technique:
            if 'smallify' in dropout_technique:
                dropout_layer = SmallifyDropout(self.layer_sizes[1], sign_variance_momentum, pruning_threshold)
            if 'variational' in dropout_technique:
                dropout_layer = VariationalDropout(self.layer_sizes[1], variational_init_droprate, pruning_threshold)

        for ndx in range(self.n_layers):
            layer_in = self.layer_sizes[ndx]
            layer_out = self.layer_sizes[ndx + 1]
            if ndx != self.n_layers - 1:

                if not use_resnet:  # M: add option to skip residual blocks
                    self.net_layers.append(SineLayer(layer_in, layer_out, bias=True, is_first=True,
                                                     dropout_layer=dropout_layer))
                    continue

                if ndx == 0:
                    # M: first layer
                    self.net_layers.append(SineLayer(layer_in, layer_out, bias=True, is_first=True,
                                                     dropout_layer=None))
                else:
                    # M: intermed layers
                    self.net_layers.append(ResidualSineBlock(self.layer_sizes[1], layer_out, bias=True,
                                                                dropout_layer=dropout_layer, ave_first=ndx > 1,
                                                                ave_second=ndx == (self.n_layers - 2)
                                                             ))
            else:
                # M: final layer
                if use_resnet:
                    final_linear = nn.Linear(self.layer_sizes[1], layer_out)
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
