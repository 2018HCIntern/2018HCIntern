import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as tcuda
from torch.autograd import Variable

class DAGAN(GAN):
    """

    """
    class Generator(nn.Module):
        def __init__(self, z_size, x_size):
            super().__init__()
            self.fc1 = nn.Linear(z_size, 1024)
            self.fc2 = nn.Linear(self.fc1.out_features, ...)
            self.fc2_bn = nn.BatchNorm1d(self.fc2.out_features)


        def forward(self, z):
            z = F.tanh(self.fc1(z))
            z = F.tanh(self.fc2_bn(self.fc2(z)))

    class Discriminator(nn.Module):
        def __init__(self, x_size, y_size):
            super().__init__()
            self.

        def forward(self, z):