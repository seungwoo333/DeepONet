import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from model import DeepONet_RNN
from dataset import GetTrainDataset, GetSuperposition, x, t, N
from dataset import AddLinearBias

num_sensors = len(x)
k = len(t)
branch_in_size = num_sensors
branch_hidden_size = 100
branch_num_layers = 2
trunk_size_ary = [1, 100, branch_hidden_size]

model = DeepONet_RNN(branch_in_size, branch_hidden_size, branch_num_layers, trunk_size_ary)

num_epoch = 20
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = StepLR(optimizer, step_size=100, gamma=0.3)
scheduler = None
criterion = nn.MSELoss()


def train(branch_input, trunk_input, target):
    branch_input = branch_input.to(device)
    trunk_input = trunk_input.to(device)
    target = target.to(device)
    
    model.init_hidden()
    out_sequence = model(branch_input, trunk_input)
    loss = criterion(out_sequence, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return torch.mean(loss.view(-1)).item()


def test(branch_input, trunk_input, target):
    branch_input = branch_input.to(device)
    trunk_input = trunk_input.to(device)
    target = target.to(device)
    
    with torch.no_grad():
        model.init_hidden()
        out_sequence = model(branch_input, trunk_input)
        loss = criterion(out_sequence, target)
        
    return torch.mean(loss.view(-1)).item()


if __name__ == "__main__":
    train_loss_ary = []
    test_loss_ary = []
    num_sequence = 100
    train_weights_ary = torch.rand(num_sequence, N)
    dataset = GetTrainDataset(train_weights_ary)

    for epoch in range(num_epoch):
        shuffled_idx = np.random.permutation(num_sequence)
        train_loss_avg = 0
        for idx in shuffled_idx:
            train_weights = torch.rand(N)
            branch_input, trunk_input, target = dataset[idx]
            avg_loss_seq = train(AddLinearBias(branch_input.to(device)), trunk_input.to(device), target.to(device))
            train_loss_avg += avg_loss_seq
            #print(f"Loss for {n}th orthogonal basis: {avg_loss_seq:.8f}")
        train_loss_avg /= num_sequence
        train_loss_ary.append(train_loss_avg)
        
        if scheduler:
            scheduler.step()
            
        test_weights = torch.rand(N)
        branch, trunk, target = GetSuperposition(test_weights)
        test_loss = test(AddLinearBias(branch.to(device)), trunk.to(device), target.to(device))
        test_loss_ary.append(test_loss)

        print(f"Epoch {epoch+1} train aveage loss: {train_loss_avg}")
        print(f"test loss by random superposition: {test_loss}")

    #torch.save(model.state_dict(), 'DeepONet_RNN_randomfield_state.pth')
    torch.save(model, './results/DeepONet_RNN_randomfield_test.pth')

    import pickle
    with open('./results/train_loss_log_test.pickle', 'wb') as f:
        pickle.dump(train_loss_ary, f, pickle.HIGHEST_PROTOCOL)
    with open('./results/test_loss_log_test.pickle', 'wb') as f:
        pickle.dump(test_loss_ary, f, pickle.HIGHEST_PROTOCOL)