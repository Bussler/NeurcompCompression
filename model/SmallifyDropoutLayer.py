import torch
from model.DropoutLayer import DropoutLayer
import torch.nn.utils.prune as prune


def calculte_smallify_loss(model, lambdaBetas = 5e-4, lambdaWeights = 5e-4):
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
    loss_Betas *= lambdaBetas
    loss_Weights *= lambdaWeights
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


class SmallifyDropout(DropoutLayer):

    def __init__(self, c):
        super(SmallifyDropout, self).__init__()
        self.c = c
        self.betas = torch.nn.Parameter(torch.empty(c).normal_(0, 1), requires_grad=True) # M: uniform_ or normal_
        #self.register_parameter('betas', self.betas)

    def forward(self, x):
        if self.training:
            x = x.mul(self.betas) # M: Scale by inverse of keep probability * (1 / (1 - self.betas)) -> Not needed here, since we mult betas with nw after training
        return x

    @classmethod
    def create_instance(cls, c):
        return SmallifyDropout(c)
