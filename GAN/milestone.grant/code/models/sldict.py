import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as tcuda
from torch.autograd import Variable


class MLPNet(nn.Module):
    def __init__(self, x_size, class_num):
        super().__init__()
        self.fc1 = nn.Linear(x_size, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, class_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


