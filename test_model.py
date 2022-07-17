import pickle
import torch
from model import DeepONet_RNN
import matplotlib.pyplot as plt
from dataset import AddLinearBias, GetSuperposition, x, t, num_sensor, k, N, alpha
import os
print(os.getcwd())


result_path = './results/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(result_path + 'DeepONet_RNN_randomfield_test.pth', map_location=device)

weights = torch.rand(N)
branch, trunk, label = GetSuperposition(weights)

with open(result_path + "train_loss_log_test.pickle","rb") as fr:
    train_loss_ary = pickle.load(fr)

with open(result_path + "test_loss_log_test.pickle","rb") as fr:
    test_loss_ary = pickle.load(fr)


plt.figure(1)
plt.plot(train_loss_ary, label='train loss')
plt.plot(test_loss_ary, label='test loss')
plt.legend()
plt.show()

weights = torch.rand(N)
weights /= torch.sum(weights)
branch, trunk, label = tuple(map(lambda tensor: tensor.to(device), GetSuperposition(weights)))

T_init = torch.unsqueeze(AddLinearBias(branch)[0], dim=0)
nodes = torch.unsqueeze(trunk[0], dim=0)
dt = t[1] - t[0]

class Solver:
    def __init__(self, NNmodel, nodes, T_init):
        self.model = NNmodel
        self.T = T_init    # [1, num_nodes, num_nodes] batched initial temperature distribution
        self.nodes = nodes  # [1, num_nodes, 1] batched node coordinates
        self.num_nodes = nodes.size(1)
        
    def step(self):
        dT = self.model(T_init, nodes)
        T_next = self.T[-1][0] + alpha * dt * (torch.squeeze(dT))
        self.T = torch.cat([self.T, torch.unsqueeze(T_next.repeat(self.num_nodes, 1), dim=0)], dim=0)
    
    def solve(self):
        self.model.init_hidden()
        for step in range(k-1):
            self.step()
            
    def plot(self, steps):
        plt.figure()
        for step in steps:
            plt.plot(self.nodes[0].view(-1).cpu().detach().numpy(), self.T[step][0].cpu().detach().numpy(), label=f"step {step}")
        plt.legend()


solver = Solver(model, nodes, T_init)
solver.solve()
solver.plot([i*10 for i in range(10)])
plt.show()