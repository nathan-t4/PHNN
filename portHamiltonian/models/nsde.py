import torch
import torchsde
import torch.nn as nn
from portHamiltonian.models.base_models import MLP

from functools import partial

class NSDE(nn.Module):
    """
        dy(t) = f(t, y(t), u(t)) dt + g(t, y(t), u(t)) dW(t);       y(t0) = y0

        where
        - t is the time (R)
        - y(t) is the state (R^n)
        - u(t) is the control (R^u)
        - f(t, y(t), u(t)) is the drift (R^n)
        - g(t, y(t), u(t)) is the diffusion (R^(m,n))
        - W(t) is Brownian motion (R^m)
    """

    noise_type = "general"
    sde_type = "ito"
    
    def __init__(self, scale, net_cfg, dtype, device):
        super().__init__()
        self.scale = scale
        self.dtype = dtype
        self.device = device

        self.in_dim = net_cfg["in_dim"]
        self.control_dim = net_cfg["control_dim"]
        self.brownian_size = net_cfg["brownian_size"]
        hidden_dim = net_cfg["hidden_dim"]
        self.mu = MLP(self.in_dim + self.control_dim, self.in_dim, hidden_dim, 1, activation=nn.ReLU()).to(self.device)
        self.sigma = MLP(self.in_dim + self.control_dim, self.in_dim * self.brownian_size, hidden_dim, 1, activation=nn.ReLU()).to(self.device)
    
    def __repr__(self):
        return f""
    
    def f(self, t, y, u: callable):
        control = u(t).reshape(1,-1)
        input = torch.cat((y,control), dim=1)
        return self.mu(input)

    def g(self, t, y, u: callable):
        control = u(t).reshape(1,-1)
        input = torch.cat((y,control), dim=1)
        batch_size, state_size = y.shape
        return self.sigma(input).view(batch_size, state_size, self.brownian_size)

    def solve(self, t, y0, control_input):
        """
            For Ito SDE, choose solvers ["euler", "milstein", "srk"]
            Note ["milstein", "srk"] do not support general noise
        """
        self.f = partial(self.f, u=control_input)
        self.g = partial(self.g, u=control_input)
        return torchsde.sdeint(self, y0, t, method="euler")