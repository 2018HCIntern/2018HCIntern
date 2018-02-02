import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as tcuda
from torch.autograd import Variable

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


class GAN:
    """
    z_size <-- Generator Input Size
    x_size <-- Generator Output Size and Discriminator Input Size
    y_size <-- Classifier Output Size
    """
    class Generator(nn.Module):
        def __init__(self, z_size, x_size):
            super().__init__()
            self.fc1 = nn.Linear(z_size, 256)
            self.fc2 = nn.Linear(self.fc1.out_features, 512)
            self.fc3 = nn.Linear(self.fc2.out_features, 1024)
            self.fc4 = nn.Linear(self.fc3.out_features, x_size)

        def forward(self, z):
            z = F.leaky_relu(self.fc1(z), 0.2)
            z = F.leaky_relu(self.fc2(z), 0.2)
            z = F.leaky_relu(self.fc3(z), 0.2)
            x = F.tanh(self.fc4(z))
            return x

    class Discriminator(nn.Module):
        def __init__(self, x_size, y_size):
            super().__init__()
            self.fc1 = nn.Linear(x_size, 1024)
            self.fc2 = nn.Linear(self.fc1.out_features, 512)
            self.fc3 = nn.Linear(self.fc2.out_features, 256)
            self.fc4 = nn.Linear(self.fc3.out_features, y_size)

        def forward(self, x):
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.dropout(x, 0.3)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.dropout(x, 0.3)
            x = F.leaky_relu(self.fc3(x), 0.2)
            x = F.dropout(x, 0.3)
            y = F.sigmoid(self.fc4(x))
            return y

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.train_loader = config.train_loader
        self.G = self.Generator(z_size=config.z_size, x_size=config.x_size)
        self.D = self.Discriminator(x_size=config.x_size, y_size=config.y_size)
        self.z_size = config.z_size

        # todo --> customizable
        self.BCE_loss = nn.BCELoss()

        # todo --> customizable
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=config.lrG)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=config.lrD)

    def train(self, epoch_num=10):

        for epoch in range(epoch_num):
            generator_losses = []
            discriminator_losses = []
            for x, _ in self.train_loader:
                self.D.zero_grad()
                y_real = torch.ones(self.batch_size)
                y_fake = torch.zeros(self.batch_size)

                if tcuda.is_available():
                    x, y_real, y_fake = x.cuda(), y_real.cuda(), y_fake.cuda()
                x, y_real, y_fake = Variable(x), Variable(y_real), Variable(y_fake)
                y = self.D(x)
                real_loss = self.BCE_loss(y.squeeze(), y_real)

                z = torch.rand((self.batch_size, self.z_size))

                if tcuda.is_available():
                    z = z.cuda()
                z = Variable(z)

                y = self.D(self.G(z))
                fake_loss = self.BCE_loss(y.squeeze(), y_fake)

                discriminator_loss = real_loss + fake_loss
                discriminator_loss.backward()
                discriminator_losses.append(discriminator_loss.data[0])
                self.D_optimizer.step()

                self.G.zero_grad()
                z = torch.rand((self.batch_size, self.z_size))

                y_fake = torch.ones(self.batch_size)

                if tcuda.is_available():
                    z, y_fake = z.cuda(), y_fake.cuda()
                z, y_fake = Variable(z), Variable(y_fake)

                y = self.D(self.G(z))
                generator_loss = self.BCE_loss(y.squeeze(), y_fake)
                generator_loss.backward()
                generator_losses.append(generator_loss.data[0])
                self.step = self.G_optimizer.step()

            logging.info('Training [%d:%d] D Loss %.6f, G Loss %.6f',
                         epoch + 1, epoch_num,
                         sum(generator_losses) / len(generator_losses),
                         sum(discriminator_losses) / len(discriminator_losses))

    def generate(self, gen_num=10):
        # z = torch.randn((gen_num, self.z_size))
        z = torch.rand((gen_num, self.z_size))
        if tcuda.is_available():
            z = z.cuda()
        z = Variable(z)
        self.G.eval()
        results = self.G(z)
        self.G.train()
        return pd.DataFrame(
            results.data.numpy(),
            columns=self.train_loader.dataset.df.columns.drop([self.train_loader.dataset.y_attr])
        )

    def save(self, generator_path, discriminator_path):
        torch.save(self.G.state_dict(), generator_path)
        torch.save(self.D.state_dict(), discriminator_path)
