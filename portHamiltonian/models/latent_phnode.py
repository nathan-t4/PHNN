import torch
import torch.nn as nn
from torchdiffeq import odeint
from portHamiltonian.models.matrix_parameters import MatrixParameter, SymmetricParameter, SkewSymmetricParameter
from models.base_models import MLP
from functools import partial

class LatentPHNODE(nn.Module):
    def __init__(self, scale, matrices_cfg, net_cfg, dtype, device):
        super().__init__()
        self.in_dim = net_cfg["in_dim"]
        self.control_dim = net_cfg["control_dim"]
        self._build_system_matrices(**matrices_cfg)
        self.scale = scale.reshape(-1,1)
        self.dtype = dtype
        self.device = device
        self._build_nets(net_cfg)

    def _build_system_matrices(self, J_cfg:dict, R_cfg:dict, B_cfg:dict, J=None, R=None, B=None, **kwargs):
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

    def _build_nets(self, net_cfg):
        self.RNN = net_cfg["model"] == "RNN"
        hidden_dim = net_cfg['hidden_dim']

        if self.RNN:
            self.encoder = nn.RNN(self.in_dim, self.in_dim, 1, batch_first=True).to(self.device)
            self.decoder = nn.RNN(self.in_dim, self.in_dim, 1, batch_first=True).to(self.device)
            self.encoder_h = torch.zeros((1,self.in_dim), device=self.device, requires_grad=True)
            self.decoder_h = torch.zeros((1,self.in_dim), device=self.device, requires_grad=True)
        else:
            self.encoder = MLP(self.in_dim, self.in_dim, hidden_dim, 1, activation=nn.ReLU()).to(self.device)
            self.decoder = MLP(self.in_dim, self.in_dim, hidden_dim, 1, activation=nn.ReLU()).to(self.device)

        self.quadratic_H = net_cfg["quadratic_H"]
        if self.quadratic_H:
            self.A = SymmetricParameter(self.in_dim, self.device, net_cfg, constant_parameter=True)
            self.H = lambda y : y.T @ self.A(y) @ y
        else:
            hidden_dim = net_cfg['hidden_dim']
            self.dH = MLP(self.in_dim, self.in_dim, hidden_dim, 1, activation=nn.ReLU()).to(self.device)

    def __repr__(self):
        default_J_input = 1.0
        default_R_input = torch.ones((1,2), device=self.device)
        default_B_input = 1.0
        J = self.J(None) if self.constant_J else self.J(default_J_input)
        R = self.R(None) if self.constant_R else self.R(default_R_input)
        B = self.B(None) if self.constant_B else self.B(default_B_input)
        return f"PHNODE with system matrices J = {J}, R = {R}, B = {B}"
    
    def forward(self, t, y, control_input, matrix_input, scale: bool):
        if self.RNN:
            encoder_h = torch.zeros((1,self.in_dim), device=self.device, requires_grad=True)
            z, _ = self.encoder(y, encoder_h)
        else:
            z = self.encoder(y)
        # z = torch.reshape(z.requires_grad_(True), (-1,1))
        # dH = torch.autograd.grad(self.H(z), z)[0]
        # Change batch dimension to axis 1
        dH = self.dH(z).T

        J = self.J(None) if self.constant_J else self.J(matrix_input(t))
        R = self.R(None) if self.constant_R else self.R(matrix_input(t))
        B = self.B(None) if self.constant_B else self.B(matrix_input(t))

        dz = (J - R) @ dH + B * control_input(t)

        if self.RNN:
            decoder_h = torch.zeros((1,self.in_dim), device=self.device, requires_grad=True)
            dx, _ = self.decoder(dz.T, decoder_h)
            dx = dx.T
        else:
            dx = self.decoder(dz.T).T
        
        if scale:
            dx = dx * self.scale

        return dx.T

    def solve(self, t, y0, control_input, matrix_input, method='dopri5'):
        dz = partial(self.forward, control_input=control_input, matrix_input=matrix_input, scale=False)
        return odeint(dz, y0, t, method=method)