import torch
from .optimizer import Optimizer, required

class SRGD(Optimizer):
    """ Implements Stochastic Relativistic Gradient Descent """
    def __init__(self, params, lr=required, gamma, m, c):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, gamma=gamma, m = m, c = c)
        super(SRGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SRGD, self).__setstate__(state)

[docs]    def step(self, closure=None):
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
                    

                p.data.add_(-group['lr'], d_p)

        return loss
