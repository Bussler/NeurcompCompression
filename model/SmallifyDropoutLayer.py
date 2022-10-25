import torch
from model.DropoutLayer import DropoutLayer
import torch.nn.utils.prune as prune
from typing import NamedTuple
import collections
import statistics
import numpy as np

def calculte_smallify_loss(model):
    loss_Betas = 0
    loss_Weights = 0

    model_state_dict = model.state_dict()

    with torch.no_grad():
        for name, param in model.named_parameters():  # M: W: .weight; biases: .bias; Drop Betas: .betas
            if name.endswith('.weight'):
                loss_Weights += param.norm()**2  # M: TODO better way to solve this
            if name.endswith('.bias'):
                loss_Weights += param.norm()**2
            if name.endswith('.betas'):
                loss_Betas += param.norm(p=1)
            if name.endswith('.betas_orig'):  # M: TODO: better way to ignore pruned betas?
                layer_split = name.split('.')
                mask_name = layer_split[0] + '.' + layer_split[1] + '.' + layer_split[2] + '.betas_mask'
                mask = model_state_dict[mask_name]
                loss_Betas += param.mul(mask).norm(p=1)
    return loss_Betas, loss_Weights

# M: Remove dropout layers and multiply them with the weights
def remove_smallify_from_model(model):
    for module in model.net_layers.modules():
        if isinstance(module, SmallifyDropout):
            prune.remove(module, 'betas')

    state_dict = model.state_dict()

    with torch.no_grad():
        for name, param, in model.state_dict().items():
            if name.endswith('.betas'):
                layer_split = name.split('.')
                if 'drop1' in name:
                    layer_name = layer_split[0]+'.'+layer_split[1]+'.linear_1'
                if 'drop2' in name:
                    layer_name = layer_split[0] + '.' + layer_split[1] + '.linear_2'
                state_dict[layer_name+'.weight'] = state_dict[layer_name+'.weight'].mul(param[:, None])
                state_dict[layer_name+'.bias'] = state_dict[layer_name+'.bias'].mul(param)
                state_dict.pop(name)
    return state_dict


def sign_variance_pruning_strategy(model, device, threshold=0.5):
    for module in model.net_layers.modules():
        if isinstance(module, SmallifyDropout):
            prune_mask = module.prune_betas_sign_variance(threshold, device)
            prune.custom_from_mask(module, name='betas', mask=prune_mask)


class Sign_Variance_Data(NamedTuple):
    values: collections.deque
    var_EMA: float
    pruned_already: bool


class SmallifyDropout(DropoutLayer):

    def __init__(self, number_betas, momentum=50):
        super(SmallifyDropout, self).__init__()
        self.c = number_betas
        self.betas = torch.nn.Parameter(torch.empty(number_betas).normal_(0, 1), requires_grad=True) # M: uniform_ or normal_
        self.momentum = momentum
        self.beta_sign_data, self.variance_emas, self.pruned_already = self.init_sign_variance_data()
        #self.register_parameter('betas', self.betas)


    def forward(self, x):
        if self.training:
            x = x.mul(self.betas) # M: Scale by inverse of keep probability * (1 / (1 - self.betas)) -> Not needed here, since we mult betas with nw after training
        return x


    def init_sign_variance_data(self):
        beta_signs = []
        variance_emas = []
        pruned_already = []
        for i in range(self.c):
            d = collections.deque(maxlen=self.momentum)
            if self.betas[i].item() > 0.0:
                d.append(1)
            else:
                d.append(-1)

            #item = Sign_Variance_Data(d, 0.0, False)
            beta_signs.append(d)
            variance_emas.append(0.0)
            pruned_already.append(False)
        return beta_signs, variance_emas, pruned_already


    def prune_betas_sign_variance(self, threshold, device):
        prune_mask = torch.zeros(self.c)

        with torch.no_grad():
            for i in range(self.c):
                data = self.beta_sign_data[i]

                if self.pruned_already[i]:
                    continue

                if self.betas[i].item() > 0.0:  # M: safe new data entry
                    self.beta_sign_data[i].append(1)
                else:
                    self.beta_sign_data[i].append(-1)
                #self.beta_sign_data[i].values.append(self.betas[i].item())  # M: safe new data entry

                var = np.var(data)
                smoother = 2 / (1+self.momentum)
                cur_ema = var * smoother + self.variance_emas[i] * (1-smoother)
                self.variance_emas[i] = cur_ema

                if cur_ema < threshold:
                    prune_mask[i] = 1.0
                else:
                    self.pruned_already[i] = True

        return prune_mask.to(device)


    @classmethod
    def create_instance(cls, c):
        return SmallifyDropout(c)
