import torch
import torch.nn as nn
import numpy as np
from numpy import pi, exp, sin, cos
from collections import OrderedDict

from dataset import x, t

batch_size = len(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Branch RNN input: [sequence, batch, feature] = [len(t), len(x): x at next timestep, len(x)]
# sequence: timestep, batch: y, feature: u
# sequence by iteration in train function: actual input size is [batch, feature]
# Network output: (d/dx)^2 u(y) at next timestep [batch]

class BranchNet_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.h = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        
        self.fc = nn.Linear(hidden_size, out_size)
        
    def forward(self, u):
        out, self.h = self.rnn(u, self.h)
        out = self.fc(out)
        return out
    
    def init_hidden(self):
        self.h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


class TrunkNet(nn.Module):
    def __init__(self, trunk_size_ary):
        super().__init__()
        
        LinearAct = lambda input_size, output_size: nn.Sequential(nn.Linear(input_size, output_size), nn.Tanh())
        
        layer_dict = []
        for i in range(len(trunk_size_ary)-1):
            layer_in = trunk_size_ary[i]
            layer_out = trunk_size_ary[i+1]
            layer_name = 'LinearAct' + str(i+1)
            layer_dict.append((layer_name, LinearAct(layer_in, layer_out)))
            
        self.layer = nn.Sequential(OrderedDict(layer_dict))
        
    def forward(self, y):
        return self.layer(y)

   
class DeepONet_RNN(nn.Module):
    def __init__(self, branch_in_size, branch_hidden_size, branch_num_layers, trunk_size_ary):
        super().__init__()
        self.BranchNet = BranchNet_RNN(branch_in_size, branch_hidden_size, branch_num_layers, trunk_size_ary[-1])
        self.TrunkNet = TrunkNet(trunk_size_ary)
        
    def forward(self, u, y):
        g = self.BranchNet(u)
        f = self.TrunkNet(y)
        G = torch.sum(g*f, dim=-1)
        return G
    
    def init_hidden(self):
        self.BranchNet.init_hidden()