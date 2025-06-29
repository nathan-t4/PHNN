import torch
import torch.nn as nn
from torchdiffeq import odeint
from portHamiltonian.models.matrix_parameters import MatrixParameter, SymmetricParameter, SkewSymmetricParameter
from models.base_models import MLP
from functools import partial

class PHNODE(nn.Module):
    def __init__(self, scale, matrices_cfg, net_cfg, dtype, device):
        super().__init__()
        self.scale = scale.reshape(-1,1)
        self.device = device
        self.dtype = dtype

        self.in_dim = net_cfg['in_dim']
        self.control_dim = net_cfg["control_dim"]
        self._build_system_matrices(**matrices_cfg)
        self._build_nets(net_cfg)


    def _build_system_matrices(self, J_cfg:dict=None, R_cfg:dict=None, B_cfg:dict=None, J=None, R=None, B=None, **kwargs):
        self.learn_J = J_cfg["learn_matrix"]
        self.learn_R = R_cfg["learn_matrix"]
        self.learn_B = B_cfg["learn_matrix"]

        self.constant_J = J_cfg["constant_parameter"]
        self.constant_R = R_cfg["constant_parameter"]
        self.constant_B = B_cfg["constant_parameter"]

        def initializeMatrix(mat, learn_mat, dim, mat_cfg, constant_parameter, MatType):
            if learn_mat:
                return MatType(dim, self.device, mat_cfg, constant_parameter)
            elif callable(mat):
                return lambda x : torch.as_tensor(mat(x), dtype=self.dtype, device=self.device)
            else:
                return lambda _ : torch.as_tensor(mat, dtype=self.dtype, device=self.device)
        
        self.J = initializeMatrix(J, self.learn_J, self.in_dim, J_cfg["net_cfg"], self.constant_J, SkewSymmetricParameter)
        self.R = initializeMatrix(R, self.learn_R, self.in_dim, R_cfg["net_cfg"], self.constant_R, SymmetricParameter)
        self.B = initializeMatrix(B, self.learn_B, self.control_dim, B_cfg["net_cfg"], self.constant_B, MatrixParameter)
    
    def __repr__(self):
        default_J_input = 1.0
        default_R_input = torch.ones((1,2), device=self.device)
        default_B_input = 1.0
        J = self.J(None) if self.constant_J else self.J(default_J_input)
        R = self.R(None) if self.constant_R else self.R(default_R_input)
        B = self.B(None) if self.constant_B else self.B(default_B_input)
        return f"PHNODE with system matrices J = {J}, R = {R}, B = {B}"

    def _build_nets(self, net_cfg):
        self.quadratic_H = net_cfg["quadratic_H"]
        if self.quadratic_H:
            self.A = SymmetricParameter(self.in_dim, self.device, net_cfg, constant_parameter=True) # TODO: Training doesn't work with this
            self.H = lambda y : y @ self.A(y) @ y.T
        else:
            hidden_dim = net_cfg['hidden_dim']
            self.dH =  MLP(self.in_dim, self.in_dim, hidden_dim, 1, activation=nn.ReLU()).to(self.device)
        
    def forward(self, t, y, control_input, matrix_input: callable, scale: bool):
        # y = y.requires_grad_(True)
        # dH = torch.autograd.grad(self.H(y), y)[0].T
        dH = self.dH(y).T
        
        J_input = None if self.constant_J else matrix_input(t)
        R_input = None if self.constant_R else y # TODO for mass spring...generalize
        B_input = None if self.constant_B else matrix_input(t)

        dx = (self.J(J_input) - self.R(R_input)) @ dH + self.B(B_input) * control_input(t)
        if scale:
            dx = dx * self.scale
        return dx.T.squeeze()

    def solve(self, t, x0, control_input, matrix_input):
        dx = partial(self.forward, control_input=control_input, matrix_input=matrix_input, scale=True)
        return odeint(dx, x0, t, method='dopri5') # options={'grid_points': t}) # options only required for adaptive step-size solvers