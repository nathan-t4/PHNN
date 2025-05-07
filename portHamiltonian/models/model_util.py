import torch as th
import torch.nn as nn

def factorial(x):
    return sum([i for i in range(x + 1)])

class MatrixParameter(nn.Module):
    """ Creates a (num_rows, num_cols) matrix parameter """
    def __init__(self, num_rows, num_cols, device, net_cfg:dict=None, constant_parameter:bool=True):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.device = device
        self.constant_parameter = constant_parameter
        self._create_parameters(num_rows * num_cols, net_cfg)

    def __repr__(self):
        if self.constant_parameter:
            return f"{self.forward()}"
        else:
            return f"{self.forward(th.ones((self.in_dim,), device=self.device))} when input is all ones"
    
    def _create_parameters(self, num_parameters, net_cfg):
        if self.constant_parameter:
            self.explicit_p = nn.Parameter(th.rand((num_parameters,1), device=self.device))
        else:
            self.in_dim = net_cfg["in_dim"]
            self.model = nn.Sequential(
                nn.Linear(self.in_dim, net_cfg["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(net_cfg["hidden_dim"], num_parameters)
            ).to(self.device)

    def forward(self, input:th.Tensor=None):
        if not self.constant_parameter:
            if len(input.shape) == 0:
                input = input.unsqueeze(0)
            matrix = self.model(input)
        else:
            matrix = self.explicit_p
        return matrix.reshape(self.num_rows, self.num_cols).to(self.device)

class SymmetricParameter(MatrixParameter):
    """ Creates a (dimension) square symmetric matrix parameter """
    def __init__(self, dimension, device, net_cfg:dict=None, constant_parameter:bool=True):
        super().__init__(dimension, dimension, device, net_cfg, constant_parameter)
        self.tril_ind = th.tril_indices(dimension, dimension, 0)
        self._create_parameters(factorial(dimension), net_cfg)

    def forward(self, input:th.Tensor=None):
        lower_tril = th.zeros((self.num_rows,self.num_rows), device=self.device)
        if not self.constant_parameter:
            if len(input.shape) == 0:
                input = input.unsqueeze(0)
            lower_tril[self.tril_ind] = self.model(input)
        else:
            lower_tril[self.tril_ind] = self.explicit_p
        return th.multiply(lower_tril, lower_tril.T) # TODO: this is positive-definite symmetric

class SkewSymmetricParameter(MatrixParameter):
    """ Creates a (dimension) square skew symmetric matrix parameter """
    def __init__(self, dimension, device, net_cfg:dict=None, constant_parameter:bool=True):
        super().__init__(dimension, dimension, device, net_cfg, constant_parameter)
        self.tril_ind = th.tril_indices(dimension, dimension, 0)
        self._create_parameters(factorial(dimension), net_cfg)
        
    def forward(self, input:th.Tensor=None):
        lower_tril = th.zeros((self.num_rows,self.num_rows), device=self.device)
        if not self.constant_parameter:
            if len(input.shape) == 0:
                input = input.unsqueeze(0)
            lower_tril[self.tril_ind] = self.model(input).unsqueeze(1)
        else:
            lower_tril[self.tril_ind] = self.explicit_p
        return lower_tril - lower_tril.T

