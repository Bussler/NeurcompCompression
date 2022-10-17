import torch
from model.DropoutLayer import DropoutLayer


class SmallifyDropout(DropoutLayer):

    def __init__(self, c):
        super(SmallifyDropout, self).__init__()
        self.c = c
        self.betas = torch.nn.Parameter(torch.empty(c).uniform_(0, 1), requires_grad=True) # M: TODO look if tensor needs grad here or if param is enough
        #self.register_parameter('betas', self.betas)

    def forward(self, x):
        x = x.mul(self.betas)  # M TODO: scale by inverse of keep probability : * (1 / (1 - self.betas)
        return x

    @classmethod
    def create_instance(cls, c):
        return SmallifyDropout(c)