import torch
from model.DropoutLayer import DropoutLayer
import torch.nn.utils.prune as prune
import model.SirenLayer as SirenLayer

def calculte_smallify_loss(model):
    loss_Betas = 0
    loss_Weights = 0

    for module in model.net_layers.modules():
        if isinstance(module, SmallifyDropout):
            loss_Betas += module.l1_loss()
        if isinstance(module, SirenLayer.SineLayer):
            loss_Weights += module.p_norm_loss()
        if isinstance(module, SirenLayer.ResidualSineBlock):
            loss_Weights += module.p_norm_loss()

    return loss_Betas, loss_Weights


def sign_variance_pruning_strategy(model, device, threshold=0.5):
    for module in model.net_layers.modules():
        if isinstance(module, SmallifyDropout):
            prune_mask = module.sign_variance_pruning(threshold, device)
            prune.custom_from_mask(module, name='betas', mask=prune_mask)


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

                smoother = 2 / (1 + self.momentum)
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
