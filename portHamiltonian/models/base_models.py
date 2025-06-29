import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden_layers, activation=nn.ReLU()):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation)

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.model(input)