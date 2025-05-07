import torch
import torch.nn as nn
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
            nn.Linear(hidden_dim, in_dim),
        ).to(self.device)
    
    def forward(self, t, y, control_input, scale: bool):
        input = torch.cat([y, control_input])
        return self.model(input)

    def solve_ode(self, t, y0, control_input):
        dot = partial(self.forward, control_input=control_input, scale=True)
        return odeint(dot, y0, t, method='dopri5') # options={'grid_points': t}) # options only required for adaptive step-size solvers