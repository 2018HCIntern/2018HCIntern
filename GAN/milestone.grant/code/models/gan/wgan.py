import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as tcuda
from torch.autograd import Variable

from .gan import GAN


class WGAN(GAN):
    """

    """
    class Generator(nn.Module):
        def __init__(self, z_size, x_size):
            super().__init__()
            self.fc1 = nn.Linear(z_size, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, x_size)

        def forward(self, z):
            z = F.relu(self.fc1(z))
            z = F.relu(self.fc2(z))
            z = F.relu(self.fc3(z))
            x = F.relu(self.fc4(z))
            return x

    class Discriminator(nn.Module):
        def __init__(self, x_size, y_size):
            super().__init__()
            self.fc1 = nn.Linear(x_size, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, y_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            y = F.relu(self.fc4(x))
            return y

    def __init__(
            self, train_lodaer,
            batch_size, learning_rate=2e-4,
            z_size=100, x_size=28*28, y_size=1, class_num=10):
        self.batch_size = batch_size
        self.train_loader = train_lodaer
        self.G = self.Generator(z_size=z_size, x_size=x_size)
        self.D = self.Discriminator(x_size=x_size, y_size=y_size)
        self.z_size = z_size

        # todo -->

        self.G_optimizer = optim

    def train(self, epoch_num=10):
        for epoch in range(epoch_num):
            generator_losses = []
            discriminator_losses = []

            for x, y in self.train_loader:
                self.D.zero_grad()
                batch_size = self.batch_size
                y_real = torch.ones(batch_size)
                y_fake = torch.zeros(batch_size)

                if tcuda.is_available():
                    x, y_real, y_fake = x.cuda(), y_real.cuda(), y_fake.cuda()
                x, y_real, y_fake = Variable(x), Variable(y_real), Variable(y_fake)

                y_pred = self.D()

    def save(self, generator_path, discriminator_path):
        super().save(generator_path, discriminator_path)