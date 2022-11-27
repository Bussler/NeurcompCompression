import torch
import torch.nn as nn
import numpy as np
import math
from model.DropoutLayer import DropoutLayer


def inference_variational_model(mu, sigma):
    return torch.normal(mu, sigma)


def calculate_Log_Likelyhood(loss, sigma):
    x_mu_loss = - loss
    a = 1 / (2 * (sigma ** 2))
    b = - torch.log(torch.sqrt(2 * torch.tensor(math.pi) * (sigma**2)))

    return a * x_mu_loss + b


# TODO nicht vergessen: da muss auch noch Regularisierung für nw weights dran!
def calculate_variational_dropout_loss(model, loss, sigma):
    Dkl_sum = 0.0
    for module in model.net_layers.modules():
        if isinstance(module, VariationalDropout):
            Dkl_sum += module.calculate_Dkl()

    Log_Likelyhood = calculate_Log_Likelyhood(loss, sigma)

    return Log_Likelyhood + Dkl_sum  # M: TODO: wir minimieren, also + dkl hier? Bei Maximieren wird ja Dkl abgezogen


class VariationalDropout(DropoutLayer):

    def __init__(self, number_thetas, threshold=0.9):
        super(VariationalDropout, self).__init__()
        self.c = number_thetas
        self.log_thetas = torch.nn.Parameter(torch.ones(number_thetas),
                                        requires_grad=True)  # M: TODO sollen eigentlich auf 1 initialisiert werden? oder doch torch.zeros()

        # M: log_var = 2*log_sigma; sigma^2 = exp(2*log_sigma) = theta^2 alpha
        self.log_var = torch.nn.Parameter(torch.ones(number_thetas),
                                             requires_grad=True)

        self.pruning_threshold = threshold

    @property
    def alphas(self):
        return torch.exp(self.log_var - 2.0 * self.log_thetas)

    @property
    def dropout_rates(self):
        return self.alphas / (1.0 + self.alphas)

    @property
    def sigma(self):
        return torch.exp(self.log_var / 2.0)

    def forward(self, x):
        # M: w = theta * (1+sqrt(alpha)*xi)
        # M: w = theta + sigma * xi according to Molchanov additive noise reparamerization
        thetas = torch.exp(self.log_thetas)  # M: revert the log with exp
        xi = torch.randn_like(x)  # M: draw xi from N(0,1)
        w = thetas + self.sigma * xi
        return x * w

    def calculate_Dkl(self):
        # M: constants from Molchanov variational dropout paper
        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695
        C = -k1

        log_alphas = torch.log(self.alphas)
        dkl = k1 * torch.sigmoid(k2 + k3 * log_alphas) - 0.5 * torch.log(1 + torch.pow(self.alphas, -1.0)) + C
        return torch.sum(dkl)  # M: Kevins code has sign swapped

    @classmethod
    def create_instance(cls, c, m):
        return VariationalDropout(c)
