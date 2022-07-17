# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 18:44:56 2022

@author: Admin
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from conduction_data import dataloader

class FC(nn.Module):
    def __init__(self, input_size, hidden_size_ary, output_size, activation):
        super().__init__()
        LayerDict = [('fc1', nn.Linear(input_size, hidden_size_ary[0])), ('act1', activation)]
        num_hidden = len(hidden_size_ary)
        
        for i in range(num_hidden-1):
            LayerDict.append((f'fc{i+2}', nn.Linear(hidden_size_ary[i], hidden_size_ary[i+1])))
            LayerDict.append((f'act{i+2}', activation))
            
        LayerDict.append((f'fc{num_hidden+1}', nn.Linear(hidden_size_ary[-1], output_size)))
        LayerDict.append((f'act{num_hidden+1}', activation))
        
        self.FClayer = nn.Sequential(OrderedDict(LayerDict))
    
    def forward(self, x):
        return self.FClayer(x)
            
    
class DeepONet(nn.Module):
    def __init__(self, in_branch, in_trunk, hidden_branch, hidden_trunk, out_size):
        super().__init__()
        act = nn.Tanh()
        self.BranchNet = FC(in_branch, hidden_branch, out_size, act)
        self.TrunkNet = FC(in_trunk, hidden_trunk, out_size, act)
        
    def forward(self, u, y):
        b = self.BranchNet(u)
        t = self.TrunkNet(y)
        out = torch.sum(b*t, dim=1)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

in_branch = 51
in_trunk = 2
hidden_branch = [100, 100, 100]
hidden_trunk = [100, 100, 100]
out_size = 100

lr = 0.0001
num_epoch = 1000

model = DeepONet(in_branch, in_trunk, hidden_branch, hidden_trunk, out_size).to(device)
model.load_state_dict(torch.load('DeepONet_1000epochs.pth'))
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

#len dataloader (number of batches) = 6*3*3*3*51
loss_ary = []
for epoch in range(num_epoch):
    print(f"-----Epoch {epoch+1}-----")
    for i, (branch_input, trunk_input, label) in enumerate(dataloader):
        branch_input = branch_input.to(device)
        trunk_input = trunk_input.to(device)
        label = label.to(device)
        out = model(branch_input, trunk_input)
        loss = criterion(out, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1000 == 0:
            print(f"Batch Step: {i+1} / {3*3*3*3*51} Current Loss: {loss.item():.4f}")
            loss_ary.append(loss.item())
    scheduler.step()