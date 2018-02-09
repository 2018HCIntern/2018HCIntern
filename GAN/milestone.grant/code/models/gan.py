import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as tcuda
from torch.autograd import Variable

import pandas as pd

from models.generators import BaseGenerator
from models.discriminators import BaseDiscriminator

from tqdm import tqdm


class GAN(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_loader = kwargs['train_loader']
        self.G = BaseGenerator(**kwargs)
        self.D = BaseDiscriminator(**kwargs)
        self.z_size = kwargs['z_size']
        if tcuda.is_available():
            self.G, self.D = self.G.cuda(), self.D.cuda()

        # todo --> customizable
        self.BCE_loss = nn.BCELoss()

        # todo --> customizable
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=kwargs['lrG'])
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=kwargs['lrD'])


    def train(self, epoch_num=100):
        pbar = tqdm(total=epoch_num)
        for epoch in range(epoch_num):
            generator_losses = []
            discriminator_losses = []
            for x, _ in self.train_loader:
                self.D.zero_grad()
                batch_size = x.size()[0]
                y_real = torch.ones(batch_size)
                y_fake = torch.zeros(batch_size)

                if tcuda.is_available():
                    x, y_real, y_fake = x.cuda(), y_real.cuda(), y_fake.cuda()
                x, y_real, y_fake = Variable(x), Variable(y_real), Variable(y_fake)
                y = self.D(x)

                real_loss = self.BCE_loss(y.squeeze(), y_real)

                z = torch.rand((batch_size, self.z_size))

                if tcuda.is_available():
                    z = z.cuda()
                z = Variable(z)

                y = self.G(z)
                y = self.D(y)
                fake_loss = self.BCE_loss(y.squeeze(), y_fake)

                discriminator_loss = real_loss + fake_loss
                discriminator_loss.backward()
                discriminator_losses.append(discriminator_loss.data[0])
                self.D_optimizer.step()

                self.G.zero_grad()
                z = torch.rand((batch_size, self.z_size))

                y_fake = torch.ones(batch_size)

                if tcuda.is_available():
                    z, y_fake = z.cuda(), y_fake.cuda()
                z, y_fake = Variable(z), Variable(y_fake)

                y = self.D(self.G(z))
                generator_loss = self.BCE_loss(y.squeeze(), y_fake)
                generator_loss.backward()
                generator_losses.append(generator_loss.data[0])
                self.step = self.G_optimizer.step()

            tqdm.write('Training [{:>5}:{:>5}] D Loss {:.6f}, G Loss {:.6f}'.format(
                     epoch + 1, epoch_num,
                     sum(generator_losses) / len(generator_losses),
                     sum(discriminator_losses) / len(discriminator_losses)))
            pbar.update(1)


    def generate(self, gen_num=10):
        z = torch.rand((gen_num, self.z_size))
        if tcuda.is_available():
            z = z.cuda()
        z = Variable(z)
        self.G.eval()
        results = self.G(z)
        self.G.train()
        return pd.DataFrame(
            results.data.numpy(),
            columns=self.train_loader.dataset.df.columns.drop([self.train_loader.dataset.y_label])
        )


    def save(self, generator_path, discriminator_path):
        torch.save(self.G.state_dict(), generator_path)
        torch.save(self.D.state_dict(), discriminator_path)

    def draw(self):
        pass
