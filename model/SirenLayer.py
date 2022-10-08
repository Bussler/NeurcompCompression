import torch
import torch.nn as nn
import numpy as np


# M: taken from https://github.com/vsitzmann/siren
class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


# M: SIREN with residual connections according to Neurcomp https://github.com/matthewberger/neurcomp
class ResidualSineLayer(nn.Module):

    def __init__(self, num_features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.num_features = num_features
        self.linear_1 = nn.Linear(num_features, num_features, bias=bias)
        self.linear_2 = nn.Linear(num_features, num_features, bias=bias)

        self.weight_1 = .5 if ave_first else 1 # M: TODO: correct like this?
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.num_features) / self.omega_0,
                                          np.sqrt(6 / self.num_features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.num_features) / self.omega_0,
                                          np.sqrt(6 / self.num_features) / self.omega_0)


    # M: concat residual block according to paper with two linear layers
    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1 * input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2 * (input+sine_2)
