import torch
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial

class NODE(nn.Module):
    def __init__(self, scale, net_cfg, dtype, device):
        super().__init__()
        self.scale = scale
        self.dtype = dtype
        self.device = device
        self._build_nets(net_cfg)
    
    def _build_nets(self, net_cfg):
        in_dim = net_cfg['in_dim']
        hidden_dim = net_cfg['hidden_dim']
        control_dim = net_cfg['control_dim']
        self.model = nn.Sequential(
            nn.Linear(in_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        ).to(self.device)
    
    def forward(self, t, y, control_input, scale: bool):
        if len(control_input.shape) <= 1:
            control_input = control_input.reshape(1,-1).to(self.device)
        input = torch.cat([y, control_input], dim=1)
        return self.model(input)

    def solve(self, t, y0, control_input):
        dot = partial(self.forward, control_input=control_input, scale=True)
        return odeint(dot, y0, t, method='dopri5') # options={'grid_points': t}) # options only required for adaptive step-size solvers