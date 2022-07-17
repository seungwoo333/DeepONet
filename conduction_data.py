# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 20:39:22 2022

@author: Admin
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import random

alpha = 0.1

num_sensor = 51
num_timestep = 20
x = np.linspace(0, 1, num_sensor)
t = np.linspace(0, 1, num_timestep)
dx = x[1]-x[0]
dt = t[1]-t[0]

# return train data at fixed location y
# input size: branch[batch_size=num_timestep, len(x)], trunk[batch_size=num_timestep, 2]
def GenerateTrainData(y, u0, u1, ua, n):
    linear_base = + u0 *(1-x) + u1*(x)
    u_init = ua*np.sin(n*pi*x)
    T = np.exp(-alpha * ((n*pi)**2) * t)
    
    u = np.array([u_init * time_exp + linear_base for time_exp in T])
    dudx_L = np.array([time_exp * n*pi*ua + (u1-u0) for time_exp in T]).reshape(-1, 1)
    dudx2 = np.array([time_exp * (-(n*pi)**2) * np.sin(n*pi*y) for time_exp in T])
    
    branch_input = torch.from_numpy(u).type(torch.float32)
    trunk_input = torch.cat([torch.from_numpy(dudx_L), y*torch.ones(num_timestep, 1)], dim=1).type(torch.float32)
    label = torch.from_numpy(dudx2).type(torch.float32)
    
    return u, branch_input, trunk_input, label


def RandomU(n):
    u0 = random.random()
    u1 = random.random()
    ua = random.random()
    y = random.random()
    return GenerateTrainData(y, u0, u1, ua, n)

def plot(u, fig=1):
    plt.figure(fig)
    for i in range(10):
        plt.plot(x, u[i*2])
    plt.show()
    
def TrainDataLoader():
    for n in range(1, 4):
        for ua in np.linspace(0.5, 1, 3):
            for u0 in np.linspace(0, 1, 3):
                for u1 in np.linspace(0, 1, 3):
                    for y in np.linspace(0, 1, 51):
                        yield GenerateTrainData(y, u0, u1, ua, n)

branchinput_full_data = torch.cat([data[1] for data in TrainDataLoader()], dim=0)
trunkinput_full_data = torch.cat([data[2] for data in TrainDataLoader()], dim=0)
label_full_data = torch.cat([data[3] for data in TrainDataLoader()], dim=0)

class ConductionDataset(Dataset):
  def __init__(self, branch_in, trunk_in, label):
    self.data = (branch_in, trunk_in, label)

  def __getitem__(self, idx):
    return self.data[0][idx], self.data[1][idx], self.data[2][idx]

  def __len__(self):
    return len(self.data[0])

dataset = ConductionDataset(branchinput_full_data, trunkinput_full_data, label_full_data)
dataloader = DataLoader(dataset, batch_size = 20, shuffle=True)


if __name__ == "__main__":
    u, branch_input, trunk_input, label = RandomU(3)
    plot(u, 1)