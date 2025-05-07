import torch
import torch.nn as nn
from torchdiffeq import odeint
from models.model_util import MatrixParameter, SymmetricParameter, SkewSymmetricParameter
from functools import partial

class LatentPHNODE(nn.Module):
    def __init__(self, scale, matrices_cfg, net_cfg, device):
        super().__init__()
        self.in_dim = net_cfg["in_dim"]
        self.control_dim = net_cfg["control_dim"]
        self._build_system_matrices(**matrices_cfg)
        self.scale = scale.reshape(-1,1)
        self.device = device
        self._build_nets(net_cfg)

    def _build_system_matrices(self, J_cfg:dict, R_cfg:dict, B_cfg:dict, **kwargs):
        self.constant_J = J_cfg["constant_parameter"]
        self.constant_R = R_cfg["constant_parameter"]
        self.constant_B = B_cfg["constant_parameter"]
        self.J = SkewSymmetricParameter(self.in_dim, self.device, J_cfg["net_cfg"], self.constant_J)
        self.R = SymmetricParameter(self.in_dim, self.device, R_cfg["net_cfg"], self.constant_R)
        self.B = MatrixParameter(self.in_dim, self.control_dim, self.device, B_cfg["net_cfg"], self.constant_B)

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

    def forward(self, t, y, control_input, matrix_input, scale: bool):
        z = self.encoder(y)
        dH = self.grad_H(z).T
        J = self.J(matrix_input(t)) if self.constant_J else self.J()
        R = self.R(matrix_input(t)) if self.constant_R else self.R()
        B = self.B(matrix_input(t))if self.constant_B else self.B()
        dz = (J - R) @ dH + B * control_input
        if scale:
            dz = dz * self.scale
        return dz.T.squeeze()

    def solve_ode(self, t, y0, control_input, matrix_input, method='dopri5'):
        dz = partial(self.forward, control_input=control_input, matrix_input=matrix_input, scale=True)
        return self.decoder(odeint(dz, y0, t, method=method))