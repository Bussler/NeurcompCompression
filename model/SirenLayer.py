import torch
import torch.nn as nn
import numpy as np
from model.DropoutLayer import DropoutLayer
from model.SmallifyDropoutLayer import SmallifyDropout


# M: taken from https://github.com/vsitzmann/siren
class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30,
                 dropout_layer: DropoutLayer = None):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=bias)

        self.drop1 = None
        if dropout_layer is not None:
           self.drop1 = dropout_layer.create_instance(out_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear_1.weight.uniform_(-1 / self.in_features,
                                              1 / self.in_features)
            else:
                self.linear_1.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                              np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        sine = self.linear_1(input)
        if self.drop1 is not None:
            sine = self.drop1(sine)

        return torch.sin(self.omega_0 * sine)


# M: SIREN with residual connections
class ResidualSineBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, ave_first=False, ave_second=False, omega_0=30,
                 dropout_layer: DropoutLayer = None):
        super().__init__()
        self.omega_0 = omega_0

        self.num_features = in_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=bias) # M: Each Res Block has two layers
        self.linear_2 = nn.Linear(out_features, out_features, bias=bias)

        self.weight_1 = .5 if ave_first else 1 # M: TODO: correct like this?
        self.weight_2 = .5 if ave_second else 1

        self.drop1 = None
        #self.drop2 = None
        if dropout_layer is not None:
            self.drop1 = dropout_layer.create_instance(out_features)
            #self.drop2 = dropout_layer.create_instance(num_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.num_features) / self.omega_0,
                                          np.sqrt(6 / self.num_features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.num_features) / self.omega_0,
                                          np.sqrt(6 / self.num_features) / self.omega_0)


    # M: concat residual block according to paper with two linear layers
    def forward(self, input):
        sine_1 = self.linear_1(self.weight_1 * input)
        if self.drop1 is not None:
            sine_1 = self.drop1(sine_1)

        sine_1 = torch.sin(self.omega_0 * sine_1)

        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        sine_2 = self.weight_2 * (input+sine_2)
        #if self.drop2 is not None:
        #    sine_2 = self.drop2(sine_2)

        return sine_2
