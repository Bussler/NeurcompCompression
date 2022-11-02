import torch
import torch.nn as nn
from torch.nn.functional import linear
import numpy as np
from model.DropoutLayer import DropoutLayer
import torch.nn.utils.prune as prune
import model.SirenLayer as SirenLayer

def calculte_smallify_loss(model):
    loss_Betas = 0
    loss_Weights = 0

    for module in model.net_layers.modules():
        if isinstance(module, SirenLayer.SineLayer):
            loss_Weights += module.p_norm_loss()
        if isinstance(module, SirenLayer.ResidualSineBlock):
            loss_Weights += module.p_norm_loss()
        if isinstance(module, SmallifyResidualSiren):
            loss_Weights += module.p_norm_loss()
            loss_Betas += module.l1_loss_betas()
        if isinstance(module, SmallifyDropout):
            loss_Betas += module.l1_loss()

    return loss_Betas, loss_Weights


def sign_variance_pruning_strategy_OD(model, device, threshold=0.5):
    for module in model.net_layers.modules():
        if isinstance(module, SmallifyDropout):
            prune_mask = module.sign_variance_pruning(threshold, device)
            prune.custom_from_mask(module, name='betas', mask=prune_mask)


def sign_variance_pruning_strategy(model, optimizer, device, threshold=0.4):
    pruned_something = False

    for module in model.net_layers.modules():
        if isinstance(module, SmallifyResidualSiren):
            prune_mask = module.sign_variance_pruning(threshold, device)
            #prune_mask = module.prune_dropout_threshold(device, 0.1)
            pruned = module.garbage_collect(prune_mask)
            if pruned:
                pruned_something = True

    if pruned_something:  # M: TODO better way to remove elements from optimizer?
        lr_list = []
        print("--CHANGING OPTIM--")
        for param_group in optimizer.param_groups:
            lr_list.append(param_group['lr'])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[0])
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] =lr_list[index]


class SmallifyDropout(DropoutLayer):

    def __init__(self, number_betas, momentum=50):
        super(SmallifyDropout, self).__init__()
        self.c = number_betas
        self.betas = torch.nn.Parameter(torch.empty(number_betas).normal_(0, 1),
                                        requires_grad=True)  # M: uniform_ or normal_
        self.momentum = momentum
        self.oldVariances, self.oldMeans, self.variance_emas, self.pruned_already, self.n = self.init_variance_data()


    def forward(self, x):
        if self.training:
            x = x.mul(self.betas)  # M: No inverse scaling needed here, since we mult betas with nw after training
        return x

    def l1_loss(self):
        return torch.abs(self.betas).sum()

    def init_variance_data(self):
        variances = []
        means = []
        variance_emas = []
        pruned_already = []
        n = 1.0

        for b in self.betas:
            if b.item() > 0.0:
                means.append(1.0)
            else:
                means.append(-1.0)

            variances.append(0.0)
            variance_emas.append(0.0)
            pruned_already.append(False)
        return variances, means, variance_emas, pruned_already, n

    def sign_variance_pruning(self, threshold, device):
        prune_mask = torch.zeros(self.c)

        with torch.no_grad():
            for i in range(self.c):

                if self.pruned_already[i]:
                    continue

                if self.betas[i].item() > 0.0:  # M: save new data entry
                    newVal = 1.0
                else:
                    newVal = -1.0

                self.oldVariances[i] = (self.n / (self.n + 1)) * (self.oldVariances[i]
                                                                  + (((self.oldMeans[i] - newVal) ** 2) / (self.n + 1)))
                self.oldMeans[i] = self.oldMeans[i] + ((newVal - self.oldMeans[i]) / (self.n + 1))

                smoother = 0.5
                self.variance_emas[i] = self.oldVariances[i] * smoother + self.variance_emas[i] * (1 - smoother)

                if self.variance_emas[i] < threshold:
                    prune_mask[i] = 1.0
                else:
                    self.pruned_already[i] = True

        self.n += 1.0
        return prune_mask.to(device)

    @classmethod
    def create_instance(cls, c):
        return SmallifyDropout(c)


class SmallifyResidualSiren(nn.Module):

    def __init__(self, in_features, intermed_features, bias=True, ave_first=False, ave_second=False, omega_0=30,
                 momentum=0.5, dropout_technique=''):
        super().__init__()
        self.omega_0 = omega_0

        self.num_features = in_features

        self.linear_weights_1 = torch.nn.Parameter(torch.empty(in_features, intermed_features).uniform_
                                                   (-np.sqrt(6 / self.num_features) / self.omega_0,
                                                    np.sqrt(6 / self.num_features) / self.omega_0),
                                                    requires_grad=True)

        self.linear_weights_2 = torch.nn.Parameter(torch.empty(intermed_features, in_features).uniform_
                                                   (-np.sqrt(6 / self.num_features) / self.omega_0,
                                                    np.sqrt(6 / self.num_features) / self.omega_0),
                                                   requires_grad=True)

        self.linear_bias_1 = None
        self.linear_bias_2 = None
        if bias:
            self.linear_bias_1 = torch.nn.Parameter(torch.empty(intermed_features).uniform_
                                                   (-np.sqrt(1/in_features), np.sqrt(1/in_features)),
                                                   requires_grad=True)
            self.linear_bias_2 = torch.nn.Parameter(torch.empty(in_features).uniform_
                                                    (-np.sqrt(1/intermed_features), np.sqrt(1/intermed_features)),
                                                    requires_grad=True)

        self.weight_1 = .5 if ave_first else 1  # M: TODO: correct like this?
        self.weight_2 = .5 if ave_second else 1

        self.c = intermed_features

        if '_quant' in dropout_technique:
            self.betas = None
        else:
            self.betas = torch.nn.Parameter(torch.empty(intermed_features).normal_(0, 1),
                                            requires_grad=True)  # M: uniform_ or normal_

            self.momentum = momentum
            self.oldVariances, self.oldMeans, self.variance_emas, self.n = self.init_variance_data()

    def forward(self, input):
        sine_1 = linear(self.weight_1 * input, self.linear_weights_1, self.linear_bias_1)
        if self.training and self.betas is not None:
            # M: TODO: TEST if dropout correct
            sine_1 = sine_1.mul(self.betas)

        sine_1 = torch.sin(self.omega_0 * sine_1)
        sine_2 = torch.sin(self.omega_0 * linear(sine_1, self.linear_weights_2, self.linear_bias_2))
        sine_2 = self.weight_2 * (input + sine_2)

        return sine_2

    def prune_dropout_threshold(self, device, threshold=0.1):
        prune_mask = (torch.abs(self.betas) > threshold).float()
        return prune_mask

    # M: dynamically change the size of linear layers according to dropout betas
    def garbage_collect(self, pruning_mask):

        # M: Stop here in case of no pruning
        if torch.count_nonzero(pruning_mask, dim=0) == pruning_mask.size()[0]:
            return False

        # M: mult betas and mask to get pruned betas
        pruned_betas = self.betas.mul(pruning_mask)
        # M: delete pruned rows
        resized_betas = pruned_betas[pruning_mask > 0]

        resized_weights_1 = self.linear_weights_1[pruning_mask > 0, :]
        resized_biases_1 = self.linear_bias_1[pruning_mask > 0]

        nonzero_indices_2 = pruning_mask.nonzero().squeeze(1)
        resized_weights_2 = torch.index_select(self.linear_weights_2, 1, nonzero_indices_2)

        # M: save new data
        new_betas = torch.nn.Parameter(resized_betas, requires_grad=True)
        self.betas = new_betas

        self.linear_weights_1 = torch.nn.Parameter(resized_weights_1, requires_grad=True)
        self.linear_bias_1 = torch.nn.Parameter(resized_biases_1, requires_grad=True)
        self.linear_weights_2 = torch.nn.Parameter(resized_weights_2, requires_grad=True)

        ndarray_mask = pruning_mask.cpu()
        self.oldVariances = self.oldVariances[ndarray_mask > 0]
        self.oldMeans = self.oldMeans[ndarray_mask > 0]
        self.variance_emas = self.variance_emas[ndarray_mask > 0]
        self.c = self.oldVariances.size

        return True

    # M: multiply first linear layer with dropout betas so that we do not have to save the betas
    def remove_dropout_layers(self):
        multiplied_weights_1 = self.linear_weights_1.mul(self.betas[:, None])
        multiplied_biases_1 = self.linear_bias_1.mul(self.betas)
        self.linear_weights_1 = torch.nn.Parameter(multiplied_weights_1, requires_grad=True)
        self.linear_bias_1 = torch.nn.Parameter(multiplied_biases_1, requires_grad=True)
        self.betas = None
        return self.c

    def p_norm_loss(self):
        lin1 = torch.sqrt(torch.abs(self.linear_weights_1).sum() ** 2) +\
               torch.sqrt(torch.abs(self.linear_bias_1).sum() ** 2)
        lin2 = torch.sqrt(torch.abs(self.linear_weights_2).sum() ** 2) +\
               torch.sqrt(torch.abs(self.linear_bias_2).sum() ** 2)
        return lin1 + lin2

    def l1_loss_betas(self):
        return torch.abs(self.betas).sum()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def init_variance_data(self):
        variances = np.zeros(shape=(self.c))
        means = np.zeros(shape=(self.c))
        variance_emas = np.zeros(shape=(self.c))
        n = 1.0

        for i in range(self.c):
            b = self.betas[i]
            if b.item() > 0.0:
                means[i] = 1.0
            else:
                means[i] = -1.0
        return variances, means, variance_emas, n

    def sign_variance_pruning(self, threshold, device):
        prune_mask = torch.zeros(self.c)

        with torch.no_grad():
            for i in range(self.c):
                if self.betas[i].item() > 0.0:  # M: save new data entry
                    newVal = 1.0
                else:
                    newVal = -1.0

                self.oldVariances[i] = (self.n / (self.n + 1)) * (self.oldVariances[i]
                                                                  + (((self.oldMeans[i] - newVal) ** 2) / (self.n + 1)))
                self.oldMeans[i] = self.oldMeans[i] + ((newVal - self.oldMeans[i]) / (self.n + 1))

                smoother = self.momentum
                self.variance_emas[i] = self.oldVariances[i] * smoother + self.variance_emas[i] * (1 - smoother)

                if self.variance_emas[i] < threshold:
                    prune_mask[i] = 1.0

        self.n += 1.0
        return prune_mask.to(device)
