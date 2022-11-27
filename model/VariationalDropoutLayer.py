import torch
import torch.nn as nn
import numpy as np
from model.DropoutLayer import DropoutLayer


# TODO nicht vergessen: da muss auch noch Regularisierung für nw weights dran!
def calculate_variational_dropout_loss(model, loss_criterion, ground_truth):
    Dkl_sum = 0.0
    for module in model.net_layers.modules():
        if isinstance(module, VariationalDropout):
            Dkl_sum += module.calculate_Dkl()

    # TODO
    Log_Likelyhood = 0.0

    return Log_Likelyhood - Dkl_sum


class VariationalDropout(DropoutLayer):

    def __init__(self, number_thetas, threshold=0.9):
        super(VariationalDropout, self).__init__()
        self.c = number_thetas
        self.log_thetas = torch.nn.Parameter(torch.zeros(number_thetas),
                                        requires_grad=True)  # M: TODO sollen eigentlich auf 1 initialisiert werden?

        # M: log_var = 2*log_sigma; sigma^2 = exp(2*log_sigma) = theta^2 alpha
        self.log_var = torch.nn.Parameter(torch.zeros(number_thetas),
                                             requires_grad=True)

        self.pruning_threshold = threshold

    @property
    def alphas(self):
        return torch.exp(self.log_var - 2.0 * self.log_thetas)

    @property
    def dropout_rates(self):
        return self.alphas / (1.0 + self.alphas)

    def forward(self, x):
        # TODO shape untersuchen von x

        # M: w = theta * (1+sqrt(alpha)*xi)
        # M: w = theta + sigma * xi according to Molchanov additive noise reparamerization
        thetas = torch.exp(self.log_thetas)  # M: revert the log with exp
        xi = torch.randn_like(x)  # M: draw xi from N(0,1)
        w = thetas + torch.exp(self.log_var / 2.0) * xi
        return x * w

    def calculate_Dkl(self):
        # M: constants from Molchanov variational dropout paper
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        C = -k1

        log_alphas = torch.log(self.alphas)
        log_alphas2 = self.log_var - 2.0 * self.log_thetas  # M: TODO: gucken, dass hier das gleiche rauskommt!
        return k1 * torch.sigmoid(k2 + k3 * log_alphas) - 0.5 * torch.log(1 + torch.pow(self.alphas, -1.0)) + C

    @classmethod
    def create_instance(cls, c):
        return VariationalDropout(c)
