import torch
import torch.nn as nn
import numpy as np
import math
from model.DropoutLayer import DropoutLayer
import torch.nn.functional as F
import model.SirenLayer as SirenLayer


def calculte_weight_loss(model):
    loss_Weights = 0.0

    for module in model.net_layers.modules():
        if isinstance(module, SirenLayer.SineLayer):
            loss_Weights = loss_Weights + module.p_norm_loss(with_bias=False)
        if isinstance(module, SirenLayer.ResidualSineBlock):
            loss_Weights = loss_Weights + module.p_norm_loss(with_bias=False)

    return loss_Weights


def calculate_log_normal_dropout_loss(model, loss_criterion, predicted_volume, ground_truth_volume, log_sigma,
                                    lambda_dkl, lambda_weights, lambda_entropy):
    Dkl_sum = 0.0
    Entropy_sum = 0.0
    for module in model.net_layers.modules():
        if isinstance(module, LogNormalDropout):
            Dkl_sum = Dkl_sum + module.calculate_Dkl()
            Entropy_sum = Entropy_sum + module.calculate_Dropout_Entropy()
    # M: lambda_dkl: sizeofelements
    Dkl_sum = (lambda_dkl/predicted_volume.shape[0]) * Dkl_sum  # 0.1 (1/predicted_volume.shape[0]) * (lambda_dkl/predicted_volume.shape[0])
    Entropy_sum = lambda_entropy * Entropy_sum  # 0.00001
    weight_loss = lambda_weights * (lambda_dkl/predicted_volume.shape[0]) * calculte_weight_loss(model)  # 0.00001 lambda_weights * (lambda_dkl/predicted_volume.shape[0])

    Log_Likelyhood = loss_criterion(predicted_volume, ground_truth_volume)

    complete_loss = -(Log_Likelyhood - Dkl_sum - weight_loss)# - Entropy_sum)

    return complete_loss, Dkl_sum, Log_Likelyhood, weight_loss


def sample_truncated_normal(mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    gamma = phi(alpha) + torch.randn_like(mu) * (phi(beta) - phi(alpha))
    return torch.clip(phi_inv(torch.clip(gamma, 1e-5, 1.0 - 1e-5)) * sigma + mu, a, b)


def phi(x):
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


def phi_inv(x):
    return math.sqrt(2.0)*erfinv(2.0*x-1)


def erfinv(x):
    #return special_math.ndtri((x+1.)/2.0)/math.sqrt(2.)
    return torch.special.ndtri((x+1.)/2.0)/math.sqrt(2.)


def erfcx(x):
    """M. M. Shepherd and J. G. Laframboise,
       MATHEMATICS OF COMPUTATION 36, 249 (1981)
    """
    K = 3.75
    y = (torch.abs(x)-K) / (torch.abs(x)+K)
    y2 = 2.0*y
    (d, dd) = (-0.4e-20, 0.0)
    (d, dd) = (y2 * d - dd + 0.3e-20, d)
    (d, dd) = (y2 * d - dd + 0.97e-19, d)
    (d, dd) = (y2 * d - dd + 0.27e-19, d)
    (d, dd) = (y2 * d - dd + -0.2187e-17, d)
    (d, dd) = (y2 * d - dd + -0.2237e-17, d)
    (d, dd) = (y2 * d - dd + 0.50681e-16, d)
    (d, dd) = (y2 * d - dd + 0.74182e-16, d)
    (d, dd) = (y2 * d - dd + -0.1250795e-14, d)
    (d, dd) = (y2 * d - dd + -0.1864563e-14, d)
    (d, dd) = (y2 * d - dd + 0.33478119e-13, d)
    (d, dd) = (y2 * d - dd + 0.32525481e-13, d)
    (d, dd) = (y2 * d - dd + -0.965469675e-12, d)
    (d, dd) = (y2 * d - dd + 0.194558685e-12, d)
    (d, dd) = (y2 * d - dd + 0.28687950109e-10, d)
    (d, dd) = (y2 * d - dd + -0.63180883409e-10, d)
    (d, dd) = (y2 * d - dd + -0.775440020883e-09, d)
    (d, dd) = (y2 * d - dd + 0.4521959811218e-08, d)
    (d, dd) = (y2 * d - dd + 0.10764999465671e-07, d)
    (d, dd) = (y2 * d - dd + -0.218864010492344e-06, d)
    (d, dd) = (y2 * d - dd + 0.774038306619849e-06, d)
    (d, dd) = (y2 * d - dd + 0.4139027986073010e-05, d)
    (d, dd) = (y2 * d - dd + -0.69169733025012064e-04, d)
    (d, dd) = (y2 * d - dd + 0.490775836525808632e-03, d)
    (d, dd) = (y2 * d - dd + -0.2413163540417608191e-02, d)
    (d, dd) = (y2 * d - dd + 0.9074997670705265094e-02, d)
    (d, dd) = (y2 * d - dd + -0.26658668435305752277e-01, d)
    (d, dd) = (y2 * d - dd + 0.59209939998191890498e-01, d)
    (d, dd) = (y2 * d - dd + -0.84249133366517915584e-01, d)
    (d, dd) = (y2 * d - dd + -0.4590054580646477331e-02, d)
    d = y * d - dd + 0.1177578934567401754080e+01
    result = d/(1.0+2.0*torch.abs(x))
    result = torch.where(torch.isnan(result), torch.ones_like(result), result)
    result = torch.where(torch.isinf(result), torch.ones_like(result), result)

    #negative_mask = tf.cast(tf.less(x, 0.0), torch.float32)
    negative_mask = torch.where(x < 0.0, 1.0, 0.0)
    #positive_mask = tf.cast(tf.greater_equal(x, 0.0), torch.float32)
    positive_mask = torch.where(x >= 0.0, 1.0, 0.0)
    negative_result = 2.0*torch.exp(x*x)-result
    negative_result = torch.where(torch.isnan(negative_result), torch.ones_like(negative_result), negative_result)
    negative_result = torch.where(torch.isinf(negative_result), torch.ones_like(negative_result), negative_result)
    result = negative_mask * negative_result + positive_mask * result
    return result


def snr_truncated_log_normal(mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    z = phi(beta) - phi(alpha)
    ratio = erfcx((sigma - beta) / math.sqrt(2.0)) * torch.exp((b - mu) - beta ** 2 / 2.0)
    ratio = ratio - erfcx((sigma - alpha) / math.sqrt(2.0)) * torch.exp((a - mu) - alpha ** 2 / 2.0)
    denominator = 2 * z * erfcx((2.0 * sigma - beta) / math.sqrt(2.0)) * torch.exp(2.0 * (b - mu) - beta ** 2 / 2.0)
    denominator = denominator - 2 * z * erfcx((2.0 * sigma - alpha) / math.sqrt(2.0)) * torch.exp(
        2.0 * (a - mu) - alpha ** 2 / 2.0)
    denominator = denominator - ratio ** 2
    ratio = ratio / torch.sqrt(denominator)  #M use math. if denominator is scalar
    return ratio


class LogNormalDropout(DropoutLayer):

    min_log = -20.0
    max_log = 0.0

    def __init__(self, number_thetas, init_dropout=0.5, threshold=0.9):
        super(LogNormalDropout, self).__init__(init_dropout, threshold)
        self.c = number_thetas
        self.pruning_threshold = 1.0

        self.mu = torch.nn.Parameter(torch.zeros(number_thetas), requires_grad=True)
        self.log_sigma = torch.nn.Parameter(torch.empty(number_thetas).fill_(-5.0), requires_grad=True)

    def forward(self, x):
        mu = torch.clip(self.mu, -20.0, 5.0)
        sigma = torch.exp(self.log_sigma)

        multiplicator = torch.exp(sample_truncated_normal(mu, sigma, self.min_log, self.max_log))

        return multiplicator * x  # M: not x*mult?
        #M: if not training: multiply mult with mask: multiplicator = mask*multiplicator

    def calculate_Dkl(self, kl_weight=1.0):
        mu = torch.clip(self.mu, -20.0, 5.0)
        sigma = torch.exp(self.log_sigma)

        # adding loss
        alpha = (self.min_log-mu)/sigma
        beta = (self.max_log-mu)/sigma
        z = phi(beta) - phi(alpha)

        def pdf(x):
            return torch.exp(-x * x / 2.0) / torch.sqrt(2.0 * np.pi)

        kl = -self.log_sigma - torch.log(z) - (alpha * pdf(alpha) - beta * pdf(beta)) / (2.0 * z)
        kl = kl + math.log(self.max_log - self.min_log) - math.log(2.0 * np.pi * np.e) / 2.0
        kl = kl_weight * torch.sum(kl)
        return kl


    def calculate_mask(self):
        mu = torch.clip(self.mu, -20.0, 5.0)
        sigma = torch.exp(self.log_sigma)
        snr = snr_truncated_log_normal(mu, sigma, self.min_log, self.max_log)
        #mask = tf.cast(tf.greater(snr, self.pruning_threshold * torch.ones_like(snr)), torch.float32)  #M: return where snr>threshold
        mask = torch.where(snr > self.pruning_threshold, 1.0, 0.0)
        return mask

    @classmethod
    def create_instance(cls, c, init_dropout=0.5, threshold=0.9):
        return LogNormalDropout(c, init_dropout, threshold)