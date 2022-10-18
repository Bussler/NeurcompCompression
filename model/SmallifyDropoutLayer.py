import torch
from model.DropoutLayer import DropoutLayer


def calculte_smallify_loss(model, lambdaBetas = 5e-4, lambdaWeights = 5e-4):
    loss_Betas = 0
    loss_Weights = 0

    for name, param in model.named_parameters():  # M: W: .weight; biases: .bias; Drop Betas: .betas
        if name.endswith('.weight'):
            loss_Weights += param.norm()
        if name.endswith('.bias'):
            loss_Weights += param.norm()
        if name.endswith('.betas'):
            loss_Betas += param.norm(p=1)
    loss_Betas *= lambdaBetas
    loss_Weights *= lambdaWeights
    return loss_Betas, loss_Weights


class SmallifyDropout(DropoutLayer):

    def __init__(self, c):
        super(SmallifyDropout, self).__init__()
        self.c = c
        self.betas = torch.nn.Parameter(torch.empty(c).uniform_(0, 1), requires_grad=True)
        #self.register_parameter('betas', self.betas)

    def forward(self, x):
        x = x.mul(self.betas)  # M: Scale by inverse of keep probability * (1 / (1 - self.betas) -> Not needed here, since we mult betas with nw after training
        return x

    @classmethod
    def create_instance(cls, c):
        return SmallifyDropout(c)
