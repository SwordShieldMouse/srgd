import torch
import math
from torch.optim.optimizer import Optimizer, required

class SRGD(Optimizer):
    """ Implements Stochastic Relativistic Gradient Descent """
    def __init__(self, params, lr=required, gamma=0.5, m=0.1, c=3e7):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, gamma=gamma, m = m, c = c)
        super(SRGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SRGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            gamma = group['gamma']
            m = group["m"]
            c = group["c"]

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                state = self.state[p]

                # initialize state
                if len(state) == 0:
                    state["p"] = torch.zeros_like(p.data)

                state["p"] = math.exp(- group["gamma"] * group["lr"]) * state["p"] - group["lr"] * d_p

                p.data.add_(group['lr'] * group["c"] * state["p"] / torch.sqrt(torch.sum(state["p"] * state["p"]) + math.pow(group["m"] * group["c"], 2) ))

        return loss
