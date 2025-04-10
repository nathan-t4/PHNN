import torch
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial

class PHNODE(nn.Module):
    def __init__(self, R, B, E, scale, net_cfg, device):
        super().__init__()
        self.R = R
        self.B = B
        self.E = E
        self.scale = scale.reshape(-1,1)
        self.device = device
        self._build_nets(net_cfg)

    def _build_nets(self, net_cfg):
        self.in_dim = net_cfg['in_dim']
        self.grad_H = nn.Sequential(
            nn.Linear(self.in_dim, net_cfg['hidden_dim']),
            nn.ReLU(),
            nn.Linear(net_cfg['hidden_dim'], net_cfg['hidden_dim']),
            nn.ReLU(),
            nn.Linear(net_cfg['hidden_dim'], 2),
            # nn.Flatten(start_dim=0),
        ).to(self.device)

        # for m in self.grad_H.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight, gain=1e-5)

        # # TODO: for testing
        # L = 20e-3
        # C = 20e-6
        # # self.H = lambda x : 0.5 * (x[0]**2 / L + x[1]**2 / C)
        # self.grad_H = lambda x : x * torch.tensor([1 / L, 1 / C], device=self.device) * (1 / self.scale).reshape(-1)
        # self.u = nn.Linear(1,1) # TODO remove

    def forward(self, t, y, u, scale: bool):
        # grad_H_fn = torch.func.vmap(torch.func.jacrev(self.H))
        # dH = grad_H_fn(y).reshape(-1,self.in_dim)[:,:2].unsqueeze(-1)
        # dH = self.grad_H(y).reshape(-1,2,1)
        # dH = torch.func.vmap(self.grad_H, in_dims=0, out_dims=0)(y).unsqueeze(-1)
        dH = self.grad_H(y).T

        # if scale:
        #     dH = dH * (1 / self.scale)

        J = torch.tensor([[0, -u(t)],
                          [u(t), 0]], device=self.device)
        dx = (J - self.R) @ dH + self.B * self.E

        if scale:
            dx = dx * self.scale
        
        return dx.T.squeeze()

    def solve_ode(self, t, x0, u):
        dx = partial(self.forward, u=u, scale=True)
        return odeint(dx, x0, t, method='dopri5') # options={'grid_points': t}) # options only required for adaptive step-size solvers

class PHTransformer(nn.Module):
    """ TODO: implement time series transformer and look at pytorch forecasting """
    def __init__(self):
        pass
