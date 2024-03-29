import torch
import torch.nn as nn
import numpy as np
import math
from model.DropoutLayer import DropoutLayer
import torch.nn.functional as F
import model.SirenLayer as SirenLayer


# M: better to infer directly
def inference_variational_model(mu, sigma):
    return torch.normal(mu, sigma)


def calculate_Log_Likelihood_variance(predicted_volume, ground_truth_volume, variance):
    x_mu_loss = (ground_truth_volume-predicted_volume) ** 2
    sigma = torch.exp(variance)
    a = 1 / (2 * (sigma ** 2))
    b = - (math.log(2 * np.pi) + (2 * variance)) / 2

    return a * (-x_mu_loss) + b, x_mu_loss


def calculte_weight_loss(model):
    loss_Weights = 0.0

    for module in model.net_layers.modules():
        if isinstance(module, SirenLayer.SineLayer):
            loss_Weights = loss_Weights + module.p_norm_loss(with_bias=False)
        if isinstance(module, SirenLayer.ResidualSineBlock):
            loss_Weights = loss_Weights + module.p_norm_loss(with_bias=False)

    return loss_Weights


def calculate_variational_dropout_loss(model, dataset_size, predicted_volume, ground_truth_volume, log_sigma,
                                       lambda_dkl, lambda_weights, lambda_entropy):
    Dkl_sum = 0.0
    Entropy_sum = 0.0
    for module in model.net_layers.modules():
        if isinstance(module, VariationalDropout):
            Dkl_sum = Dkl_sum + module.calculate_Dkl()
            Entropy_sum = Entropy_sum + module.calculate_Dropout_Entropy()
    Dkl_sum = lambda_dkl * (dataset_size / predicted_volume.shape[0]) * Dkl_sum
    Entropy_sum = lambda_entropy * (dataset_size / predicted_volume.shape[0]) * Entropy_sum
    weight_loss = lambda_weights * (dataset_size / predicted_volume.shape[0]) * calculte_weight_loss(model)

    # M: iter over all values
    Log_Likelyhood, mse = calculate_Log_Likelihood_variance(predicted_volume, ground_truth_volume, log_sigma)
    Log_Likelyhood = Log_Likelyhood.sum() * (dataset_size / predicted_volume.shape[0])
    mse = mse.sum() * (1/predicted_volume.shape[0])

    complete_loss = -(Log_Likelyhood - Dkl_sum - weight_loss - Entropy_sum)

    return complete_loss, Dkl_sum, Log_Likelyhood, mse, weight_loss


class VariationalDropout(DropoutLayer):
    # M: constants from Molchanov variational dropout paper
    k1 = 0.63576
    k2 = 1.87320
    k3 = 1.48695
    C = -k1

    i = 0
    theshold_list = None

    def __init__(self, number_thetas, init_dropout=0.5, threshold=0.9):
        super(VariationalDropout, self).__init__(init_dropout, threshold)
        self.c = number_thetas
        self.log_thetas = torch.nn.Parameter(torch.zeros(number_thetas), requires_grad=True)

        log_alphas = math.log(init_dropout / (1-init_dropout))
        # M: log_var = 2*log_sigma; sigma^2 = exp(2*log_sigma) = theta^2 alpha
        self.log_var = torch.nn.Parameter(torch.empty(number_thetas).fill_(log_alphas), requires_grad=True)

        self.pruning_threshold = threshold
        if VariationalDropout.theshold_list is not None:  #M used for testing, needs refactoring
            if VariationalDropout.i != 0:
                self.pruning_threshold = VariationalDropout.theshold_list[VariationalDropout.i-1]
            VariationalDropout.i = VariationalDropout.i +1

    @property
    def alphas(self):
        return torch.exp(self.log_var - 2.0 * self.log_thetas)

    @property
    def dropout_rates(self):
        return self.alphas / (1.0 + self.alphas)  #M: maybe wrong? 1-alphas?

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
        log_alphas = self.log_var - 2.0 * self.log_thetas

        t1 = self.k1 * torch.sigmoid(self.k2 + self.k3 * log_alphas)
        t2 = 0.5 * F.softplus(-log_alphas, beta=1.)
        dkl = - t1 + t2 + self.k1

        return torch.sum(dkl)

    def calculate_Dropout_Entropy(self):
        drop_rate = self.dropout_rates
        h = drop_rate * torch.log(drop_rate) + (1.0 - drop_rate) * torch.log(1 - drop_rate)
        return torch.sum(h)

    # M: If dropout_rates close to 1: alpha >> 1 and theta has no useful information
    def get_valid_thetas(self):
        dropout_rates = self.dropout_rates
        # M: find indices of thetas to remove
        indices = torch.where(dropout_rates < self.pruning_threshold, 1, 0).nonzero().squeeze(1)

        if indices.shape[0] == 0:
            return torch.ones(1).to('cuda'), torch.zeros(1, dtype=torch.int).to('cuda')

        return torch.exp(self.log_thetas)[dropout_rates < self.pruning_threshold], indices

    def get_invalid_thetas(self):
        dropout_rates = self.dropout_rates
        indices = torch.where(dropout_rates >= self.pruning_threshold, 1, 0).nonzero().squeeze(1)
        return indices

    def get_valid_fraction(self):
        not_dropped = torch.mean((self.dropout_rates < self.pruning_threshold).to(torch.float)).item()
        bin1 = torch.mean((self.dropout_rates < 0.1).to(torch.float)).item()
        bin3 = torch.mean((self.dropout_rates > 0.9).to(torch.float)).item()
        bin2 = 1.0-bin1-bin3
        return not_dropped, self.dropout_rates

    @classmethod
    def set_feature_list(cls, list: []):
        cls.theshold_list = list

    @classmethod
    def create_instance(cls, c, init_dropout=0.5, threshold=0.9):
        return VariationalDropout(c, init_dropout, threshold)
