import torch
import torch.nn as nn
import numpy as np
from model.SirenLayer import SineLayer, ResidualSineBlock
from model.SmallifyDropoutLayer import SmallifyDropout, SmallifyResidualSiren


# M: Neurcomp according to "Compressive Neural Representations of Volumetric Scalar Fields"
class Neurcomp(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, features=[], omega_0=30, dropout_technique='',
                 sign_variance_momentum=0.5):
        super(Neurcomp, self).__init__()

        self.d_in = input_ch
        self.d_out = output_ch
        self.omega_0 = omega_0  # M: scale hyperparameter for SIREN

        self.layer_sizes = [self.d_in]
        self.layer_sizes.extend(features)
        self.layer_sizes.append(self.d_out)
        self.n_layers = len(self.layer_sizes) - 1

        self.net_layers = nn.ModuleList()

        # M: setup dropout
        dropout_layer = None
        if dropout_technique and '_quant' not in dropout_technique:
            if 'smallify' in dropout_technique:
                dropout_layer = SmallifyDropout(self.layer_sizes[1])
            if dropout_technique == 'variational':
                pass

        for ndx in range(self.n_layers):
            layer_in = self.layer_sizes[ndx]
            layer_out = self.layer_sizes[ndx + 1]
            if ndx != self.n_layers - 1:
                if ndx == 0:
                    # M: first layer
                    self.net_layers.append(SineLayer(layer_in, layer_out, bias=True, is_first=True,
                                                     dropout_layer=None))  # M: TODO also try dropout_layer
                else:
                    # M: intermed layers
                    if False:#dropout_technique and 'smallify' in dropout_technique:
                        self.net_layers.append(SmallifyResidualSiren(self.layer_sizes[1], layer_out, bias=True,
                                                                     ave_first=ndx > 1,
                                                                     ave_second=ndx == (self.n_layers - 2),
                                                                     dropout_technique=dropout_technique))
                    else:
                        self.net_layers.append(ResidualSineBlock(self.layer_sizes[1], layer_out, bias=True,
                                                                 dropout_layer=dropout_layer, ave_first=ndx > 1,
                                                                 ave_second=ndx == (self.n_layers - 2)))
            else:
                final_linear = nn.Linear(self.layer_sizes[1], layer_out) #nn.Linear(layer_in, layer_out)
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
