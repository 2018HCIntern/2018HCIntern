import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as tcuda
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

import pandas as pd

from models.gan import GAN
from models.generators import ConditionalBNGenerator, ConditionalGenerator
from models.discriminators import ConditionalBNDiscriminator, ConditionalDiscriminator

from tqdm import tqdm


class CGAN(GAN):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.train_loader = kwargs['train_loader']
        self.G = ConditionalBNGenerator(**kwargs)
        # self.G = ConditionalGenerator(**kwargs)
        self.D = ConditionalBNDiscriminator(**kwargs)
        # self.D = ConditionalDiscriminator(**kwargs)
        self.z_size = kwargs['z_size']
        self.class_num = kwargs['class_num']

        if tcuda.is_available():
            self.G, self.D = self.G.cuda(), self.D.cuda()

        # todo --> customizable
        self.BCE_loss = nn.BCELoss()

        # todo --> customizable
        # todo --> weight decay setting
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=kwargs['lrG'])
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=kwargs['lrD'])
        self.G_scheduler = MultiStepLR(self.G_optimizer, milestones=[30, 40], gamma=0.1)
        self.D_scheduler = MultiStepLR(self.D_optimizer, milestones=[30, 40], gamma=0.1)


    def train(self, epoch_num=10):
        # self.G.weight_init(mean=0, std=0.02)
        # self.D.weight_init(mean=0, std=0.02)
        pbar = tqdm(total=epoch_num)
        for epoch in range(epoch_num):
            generator_losses = []
            discriminator_losses = []

            self.G_scheduler.step()
            self.D_scheduler.step()

            for x, y in self.train_loader:
                self.D.zero_grad()
                batch_size = x.size()[0]

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
                fake_loss = self.BCE_loss(y_pred, y_fake)

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

            tqdm.write('Training [{:>5}:{:>5}] D Loss {:.6f}, G Loss {:.6f}'.format(
                     epoch + 1, epoch_num,
                     sum(generator_losses) / len(generator_losses),
                     sum(discriminator_losses) / len(discriminator_losses)))
            pbar.update(1)


    def generate(self, gen_num=10):
        z = torch.rand(gen_num, self.z_size)
        c_ = torch.zeros(gen_num // self.class_num, 1)
        for i in range(1, self.class_num):
            temp = torch.zeros(gen_num // self.class_num, 1) + i
            c_ = torch.cat([c_, temp], 0)
        c = torch.zeros(gen_num, self.class_num)
        c.scatter_(1, c_.type(torch.LongTensor), 1)
        if tcuda.is_available():
            z, c = z.cuda(), c.cuda()
        z, c = Variable(z), Variable(c)
        self.G.eval()
        results = self.G(z, c)
        resultsd = torch.cat([results.data, c_], 1)
        self.G.train()
        return pd.DataFrame(
            resultsd.numpy(),
            columns=self.train_loader.dataset.df.columns
        )


    def save(self, generator_path, discriminator_path):
        super().save(generator_path, discriminator_path)
