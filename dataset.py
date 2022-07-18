import numpy as np
import torch
from numpy import pi, exp, sin, cos
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = np.linspace(0, 1, 51)
t = np.linspace(0, 10, 1001)
num_sensor = len(x)
k = len(t)

alpha = 0.1
u = lambda n, x, t: exp(-alpha * (n*pi)**2 * t) * sin(n*pi*x)
d2udx2 = lambda n, x, t: -(n*pi)**2 * u(n, x, t)

def get_functional(functional, n):
    func = lambda x, t: functional(n, x, t)
    return func

N = 3
functional_ary_u = [get_functional(u, n) for n in range(1, N+1)]
functional_ary_d2udx2 = [get_functional(d2udx2, n) for n in range(1, N+1)]


# branch input size [sequence=k, batch=num_sensors, features=num_sensors]
def GetBranchInput(n):
    u = functional_ary_u[n-1]
    branch_inputs = []
    
    for timestep in t[:k-1]:
        branch_inputs.append(torch.from_numpy(u(x, timestep)).type(torch.float32).view(1, -1))
        
    branch_inputs = torch.cat(branch_inputs, dim=0) # cat sequence
    branch_inputs = torch.stack([branch_inputs for _ in range(num_sensor)], dim=1) # expand batch
    
    return branch_inputs


# trunk input size [sequence=k, batch=num_sensors, features=1 (domain dimension)]
def GetTrunkInput(n):
    y = torch.from_numpy(x).type(torch.float32).view(-1, 1)
    trunk_inputs = []
    
    for timestep in t[:k-1]:
        trunk_inputs.append(y)
        
    trunk_inputs = torch.stack(trunk_inputs, dim=0) # cat sequence
    
    return trunk_inputs


# target size [sequence=k, batch=num_sensors]
def GetTarget(n):
    targets = []
    for timestep in t[1:]:
        targets.append(torch.from_numpy(functional_ary_d2udx2[n-1](x, timestep)).type(torch.float32))
        
    targets = torch.stack(targets, dim=0)
    
    return targets


def GetSequence(n):
    return GetBranchInput(n), GetTrunkInput(n), GetTarget(n)


def GetSuperposition(weights):
    for order, weight in enumerate(weights):
        branch, trunk, target = GetSequence(order + 1)
        
        if order == 0:
            branch_sup = torch.zeros_like(branch)
            target_sup = torch.zeros_like(target)
         
        branch_sup += branch * weight
        target_sup += target * weight
        
    return branch_sup, trunk, target_sup


def GetTrainDataset(train_weights_ary):
    dataset = []
    for weights in train_weights_ary:
        dataset.append(GetSuperposition(weights))

    return dataset


def AddLinearBias(branch_input):
    u0 = 2 * random.random() - 1
    u1 = 2 * random.random() - 1
    bias = torch.from_numpy(u0*(1-x) + u1*x).type(torch.float32).view(1, 1, num_sensor).to(device)
    branch_input += bias

    return branch_input