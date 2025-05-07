import torch
import torch.nn as nn
from torchdiffeq import odeint
from models.model_util import MatrixParameter, SymmetricParameter, SkewSymmetricParameter
from functools import partial

class PHNODE(nn.Module):
    def __init__(self, scale, matrices_cfg, net_cfg, device):
        super().__init__()
        self.scale = scale.reshape(-1,1)
        self.device = device
        self.in_dim = net_cfg['in_dim']
        self.control_dim = net_cfg["control_dim"]
        self._build_system_matrices(**matrices_cfg)
        self._build_nets(net_cfg)

    def _build_system_matrices(self, J_cfg:dict=None, R_cfg:dict=None, B_cfg:dict=None, J=None, R=None, B=None, **kwargs):
        self.learn_J = J_cfg["learn_matrix"]
        self.learn_R = R_cfg["learn_matrix"]
        self.learn_B = B_cfg["learn_matrix"]
        if self.learn_J or self.learn_R or self.learn_B:
            self.J = SkewSymmetricParameter(self.in_dim, self.device, J_cfg["net_cfg"], J_cfg["constant_parameter"]) \
                if self.learn_J else torch.as_tensor(J, dtype=torch.float32, device=self.device)
            self.R = SymmetricParameter(self.in_dim, self.device, R_cfg["net_cfg"], R_cfg["constant_parameter"]) \
                if self.learn_R else torch.as_tensor(R, dtype=torch.float32, device=self.device)
            self.B = MatrixParameter(self.in_dim, self.control_dim, self.device, B_cfg["net_cfg"], B_cfg["constant_parameter"]) \
                if self.learn_B else torch.as_tensor(B, dtype=torch.float32, device=self.device)  
        else:
            self.J = torch.as_tensor(J, dtype=torch.float32, device=self.device) if J is not None else None
            self.R = torch.as_tensor(R, dtype=torch.float32, device=self.device) if R is not None else None
            self.B = torch.as_tensor(B, dtype=torch.float32, device=self.device) if B is not None else None
    
    def __repr__(self):
        return f"PHNODE with system matrices J = {self.J}, R = {self.R}, B = {self.B}"

    def _build_nets(self, net_cfg):
        hidden_dim = net_cfg['hidden_dim']
        self.grad_H = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            # nn.Flatten(start_dim=0),
        ).to(self.device)

    def forward(self, t, y, control_input, matrix_input, scale: bool):
        dH = self.grad_H(y).T
        
        if self.learn_J or self.learn_R or self.learn_B:
            J_input = matrix_input(t) if self.learn_J else None
            R_input = matrix_input(t) if self.learn_R else None
            B_input = matrix_input(t) if self.learn_B else None
            dx = (self.J(J_input) - self.R(R_input)) @ dH + self.B(B_input) * control_input
        else:
            J = torch.tensor([[0, -matrix_input(t)],
                              [matrix_input(t), 0]], device=self.device) # TODO: generalize
            dx = (J - self.R) @ dH + self.B * control_input
        if scale:
            dx = dx * self.scale
        return dx.T.squeeze()

    def solve_ode(self, t, x0, control_input, matrix_input):
        dx = partial(self.forward, control_input=control_input, matrix_input=matrix_input, scale=True)
        return odeint(dx, x0, t, method='dopri5') # options={'grid_points': t}) # options only required for adaptive step-size solvers