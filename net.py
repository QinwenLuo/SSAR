import torch
import torch.nn as nn
import numpy as np


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class ParaNet(nn.Module):
    def __init__(self, input_size, init_value=1.0, hidden_dims=None, squeeze_output=True, last_activation_fn='sigmoid'):
        super(ParaNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        fc1_dims = hidden_dims[0]
        fc2_dims = hidden_dims[1]
        fc3_dims = hidden_dims[2]
        self.fc1 = nn.Linear(in_features=input_size, out_features=fc1_dims)
        self.fc2 = nn.Linear(in_features=fc1_dims, out_features=fc2_dims)
        self.fc3 = nn.Linear(in_features=fc2_dims, out_features=fc3_dims)
        self.fc4 = nn.Linear(in_features=fc3_dims, out_features=1)
        self.activation_fn = nn.ReLU()


        self.last_activation_fn = nn.Sigmoid()
        bias_init_value = 0.6931

        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.xavier_uniform_(self.fc3.weight.data)
        nn.init.xavier_uniform_(self.fc4.weight.data)

        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.fc4.bias.data.fill_(bias_init_value)

        self.max_value = init_value * 1.5

        if squeeze_output:
            self.squeeze = Squeeze(-1)
        else:
            self.squeeze = nn.Identity()

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.activation_fn(self.fc3(x))
        x = self.fc4(x)
        x = self.last_activation_fn(x)
        return self.squeeze(x) * self.max_value
