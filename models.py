import torch
import torch.nn as nn
import torch.nn.functional as F


class MP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.hl1 = nn.Linear(in_dim, hid_dim)
        nn.init.xavier_uniform_(self.hl1.weight)
        nn.init.zeros_(self.hl1.bias)
        self.hl2 = nn.Linear(hid_dim, out_dim)
        nn.init.xavier_uniform_(self.hl2.weight)
        nn.init.zeros_(self.hl2.bias)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.hl1(x))
        x = self.ReLU(self.hl2(x))
        return x

class CNN_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
