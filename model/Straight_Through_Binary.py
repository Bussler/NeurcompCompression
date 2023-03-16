import torch
import torch.nn as nn
from torch.nn.functional import linear
import numpy as np
from model.DropoutLayer import DropoutLayer
import torch.nn.utils.prune as prune
import model.SirenLayer as SirenLayer

def calculte_binyary_mask_loss(model):
    loss_Betas = 0
    loss_Weights = 0

    for module in model.net_layers.modules():
        if isinstance(module, SirenLayer.SineLayer):
            loss_Weights = loss_Weights + module.p_norm_loss()
        if isinstance(module, SirenLayer.ResidualSineBlock):
            loss_Weights = loss_Weights + module.p_norm_loss()
        if isinstance(module, MaskedWavelet_Straight_Through_Dropout):
            loss_Betas = loss_Betas + module.l1_loss()

    return loss_Betas, loss_Weights

class MaskedWavelet_Straight_Through_Dropout(DropoutLayer):

    def __init__(self, size=(1, 1, 1), probability=0.5, threshold=0.5):
        super(MaskedWavelet_Straight_Through_Dropout, self).__init__(probability, threshold)
        self.c = size
        #self.mask_values = torch.nn.Parameter(torch.ones(size), requires_grad=True)  # M: uniform_ or normal_ ones
        self.mask_values = torch.nn.Parameter(torch.empty(size).uniform_(0, 1), requires_grad=True)

    def forward(self, x):
        if self.training:
            mask = torch.sigmoid(self.mask_values)
            x = (x * (mask >= self.threshold) - x * mask).detach() + (x * mask)  # M: Need inverse scaling?
            #x = x * (mask >= self.threshold).detach()  # M: Need inverse scaling?
        return x

    def l1_loss(self):
        return torch.sigmoid(self.mask_values).sum()

    def calculate_pruning_mask(self):
        mask = torch.abs(self.mask_values)
        return mask

    def calculate_final_value_for_pruning(self, ):
        with torch.no_grad():
            mask = self.calculate_pruning_mask()
            #f_grid = (input * (mask >= self.threshold) - input * mask).detach() + (input * mask)

            indices = torch.where(mask >= self.threshold, 1, 0).nonzero().squeeze(1)
            if indices.shape[0] == 0:
                return torch.ones(1).to('cuda'), torch.zeros(1, dtype=torch.int).to('cuda'), torch.tensor(self.threshold).to('cuda')
            return mask[mask >= self.threshold], indices, torch.tensor(self.threshold).to('cuda')

    def size_layer(self):
        return self.mask_values.numel()

    @classmethod
    def create_instance(cls, c, sign_variance_momentum=0.02, threshold = 0.9):
        return MaskedWavelet_Straight_Through_Dropout(c, sign_variance_momentum, threshold)
