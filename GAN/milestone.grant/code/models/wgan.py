import torch.nn as nn
import torch.nn.functional as F

from models.gan import GAN


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