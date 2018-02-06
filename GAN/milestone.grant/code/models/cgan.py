import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as tcuda
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

import pandas as pd

from models.gan import GAN
from models.generators import ConditionalBNGenerator
from models.discriminators import ConditionalBNDiscriminator


class CGAN(GAN):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.train_loader = kwargs['train_loader']
        self.G = ConditionalBNGenerator(**kwargs)
        self.D = ConditionalBNDiscriminator(**kwargs)
        self.z_size = kwargs['z_size']
        self.class_num = kwargs['class_num']

        # todo --> customizable
        self.BCE_loss = nn.BCELoss()

        # todo --> customizable
        # todo --> weight decay setting
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=kwargs['lrG'])
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=kwargs['lrD'])
        self.G_scheduler = MultiStepLR(self.G_optimizer, milestones=[30, 40], gamma=0.1)
        self.D_scheduler = MultiStepLR(self.D_optimizer, milestones=[30, 40], gamma=0.1)


    def train(self, epoch_num=10):
        for epoch in range(epoch_num):
            generator_losses = []
            discriminator_losses = []

            self.G_scheduler.step()
            self.D_scheduler.step()

            for x, y in self.train_loader:
                self.D.zero_grad()
                batch_size = self.batch_size
                y_real = torch.ones(batch_size)
                y_fake = torch.zeros(batch_size)
                y_label = torch.zeros(batch_size, self.class_num)
                y_label.scatter_(1, y.view(batch_size, 1), 1)

                if tcuda.is_available():
                    x, y_real, y_fake, y_label = x.cuda(), y_real.cuda(), y_fake.cuda(), y_label.cuda()
                x, y_real, y_fake, y_label = Variable(x), Variable(y_real), Variable(y_fake), Variable(y_label)

                y_pred = self.D(x, y_label).squeeze()
                real_loss = self.BCE_loss(y_pred, y_real)

                z = torch.rand((batch_size, self.z_size))
                y_pred = (torch.rand(batch_size, 1) * self.class_num).type(torch.LongTensor)
                y_label = torch.zeros(batch_size, self.class_num)
                y_label.scatter_(1, y_pred.view(batch_size, 1), 1)

                if tcuda.is_available():
                    z, y_label = z.cuda(), y_label.cuda()
                z, y_label = Variable(z), Variable(y_label)
                y_pred = self.D(self.G(z, y_label), y_label).squeeze()
                fake_loss = self.BCE_loss(y_pred, y_real)

                train_loss = real_loss + fake_loss
                train_loss.backward()
                discriminator_losses.append(train_loss.data[0])

                self.D_optimizer.step()

                self.G.zero_grad()
                z = torch.rand((batch_size, self.z_size))
                y_pred = (torch.rand(batch_size, 1) * self.class_num).type(torch.LongTensor)
                y_label = torch.zeros(batch_size, self.class_num)
                y_label.scatter_(1, y_pred.view(batch_size, 1), 1)

                if tcuda.is_available():
                    z, y_label = Variable(z), Variable(y_label)
                z, y_label = Variable(z), Variable(y_label)
                y_pred = self.D(self.G(z, y_label), y_label).squeeze()
                train_loss = self.BCE_loss(y_pred, y_real)
                train_loss.backward()
                generator_losses.append(train_loss.data[0])

                self.G_optimizer.step()

            print('Training [%d:%d] D Loss %.6f, G Loss %.6f',
                     epoch + 1, epoch_num,
                     sum(generator_losses) / len(generator_losses),
                     sum(discriminator_losses) / len(discriminator_losses))


    def generate(self, gen_num=10):
        z = torch.rand(gen_num, self.z_size + self.class_num)
        if tcuda.is_available():
            z = z.cuda()
        z = Variable(z)
        self.G.eval()
        results = self.G(z)
        self.G.train()
        return results


    def save(self, generator_path, discriminator_path):
        super().save(generator_path, discriminator_path)
