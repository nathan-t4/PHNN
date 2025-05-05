import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchdiffeq import odeint
from functools import partial

class NODE(nn.Module):
    def __init__(self, net_cfg):
        self._build_nets(net_cfg)
    
    def _build_nets(self, net_cfg):
        in_dim = net_cfg['in_dim']
        hidden_dim = net_cfg['hidden_dim']
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            # nn.Flatten(start_dim=0),
        ).to(self.device)
    
    def forward(self, t, y, u, scale: bool):
        input = torch.cat([y, u])
        return self.model(input)

    def solve_ode(self, t, y0, u):
        dot = partial(self.forward, u=u, scale=True)
        return odeint(dot, y0, t, method='dopri5') # options={'grid_points': t}) # options only required for adaptive step-size solvers

class PHNODE(nn.Module):
    def __init__(self, scale, matrices, net_cfg, device):
        super().__init__()
        self.scale = scale.reshape(-1,1)
        self.device = device
        self._build_system_matrices(**matrices)
        self._build_nets(net_cfg)

    def _build_system_matrices(self, J=None, R=None, B=None):
        self.J = torch.as_tensor(J, dtype=torch.float32, device=self.device) if J is not None else None
        self.R = torch.as_tensor(R, dtype=torch.float32, device=self.device) if R is not None else None
        self.B = torch.as_tensor(B, dtype=torch.float32, device=self.device) if B is not None else None

    def _build_nets(self, net_cfg):
        in_dim = net_cfg['in_dim']
        hidden_dim = net_cfg['hidden_dim']
        self.grad_H = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            # nn.Flatten(start_dim=0),
        ).to(self.device)

    def forward(self, t, y, E, u, scale: bool):
        dH = self.grad_H(y).T
        J = torch.tensor([[0, -u(t)],
                          [u(t), 0]], device=self.device) # TODO: generalize
        dx = (J - self.R) @ dH + self.B * E
        if scale:
            dx = dx * self.scale
        return dx.T.squeeze()

    def solve_ode(self, t, x0, E, u):
        dx = partial(self.forward, E=E, u=u, scale=True)
        return odeint(dx, x0, t, method='dopri5') # options={'grid_points': t}) # options only required for adaptive step-size solvers

class LatentPHNODE(nn.Module):
    def __init__(self, scale, net_cfg, device):
        super().__init__()
        self.in_dim = net_cfg["in_dim"]
        self.control_dim = net_cfg["control_dim"]
        self._build_system_matrices()
        self.scale = scale.reshape(-1,1)
        self.device = device
        self._build_nets(net_cfg)

    def _build_system_matrices(self):
        random_matrix = Parameter(torch.rand(self.in_dim, self.in_dim))
        self.J = random_matrix - random_matrix.T
        upper_tri = Parameter(torch.triu(torch.rand(self.in_dim, self.in_dim)))
        self.R = torch.multiply(upper_tri, upper_tri.T)
        self.B = Parameter(torch.rand(self.in_dim, self.control_dim))

    def _build_nets(self, net_cfg):
        hidden_dim = net_cfg['hidden_dim']

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.in_dim)
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.in_dim)
        ).to(self.device)

        self.grad_H = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        ).to(self.device)

    def forward(self, t, y, E, u, scale: bool):
        z = self.encoder(y)
        dH = self.grad_H(z).T
        dz = (self.J - self.R) @ dH + self.B * E
        if scale:
            dz = dz * self.scale
        return dz.T.squeeze()

    def solve_ode(self, t, y0, E, u, method='dopri5'):
        dz = partial(self.forward, E=E, u=u, scale=True)
        return self.decoder(odeint(dz, y0, t, method=method)) # options={'grid_points': t}) # options only required for adaptive step-size solvers


class PHTransformer(nn.Module):
    """ TODO: implement time series transformer and look at pytorch forecasting """
    def __init__(self):
        pass
