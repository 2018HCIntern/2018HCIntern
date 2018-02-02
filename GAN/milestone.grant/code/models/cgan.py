import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as tcuda
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

from models.gan import GAN

import logging

logging.basicConfig(level=logging.INFO)


class CGAN(GAN):
    """
    z_size <-- Generator Input Size
    x_size <-- Generator Output Size and Discriminator Input Size
    y_size <-- Classifier Output Size
    """
    class Generator(nn.Module):
        def __init__(self, z_size, x_size, class_num):
            super().__init__()
            self.fc1_1 = nn.Linear(z_size, 256)
            self.fc1_1_bn = nn.BatchNorm1d(self.fc1_1.out_features)
            self.fc1_2 = nn.Linear(class_num, 256)
            self.fc1_2_bn = nn.BatchNorm1d(self.fc1_2.out_features)
            self.fc2 = nn.Linear(self.fc1_1.out_features + self.fc1_2.out_features, 512)
            self.fc2_bn = nn.BatchNorm1d(self.fc2.out_features)
            self.fc3 = nn.Linear(self.fc2.out_features, 1024)
            self.fc3_bn = nn.BatchNorm1d(self.fc3.out_features)
            self.fc4 = nn.Linear(self.fc3.out_features, x_size)

        def weight_init(self, mean, std):
            for m in self._modules:
                CGAN.normal_init(self._modules[m], mean, std)

        def forward(self, z, c):
            z_1 = F.relu(self.fc1_1_bn(self.fc1_1(z)))
            z_2 = F.relu(self.fc1_2_bn(self.fc1_2(c)))
            z = torch.cat([z_1, z_2], 1)
            z = F.relu(self.fc2_bn(self.fc2(z)))
            z = F.relu(self.fc3_bn(self.fc3(z)))
            x = F.tanh(self.fc4(z))
            return x

    class Discriminator(nn.Module):
        def __init__(self, x_size, y_size, class_num):
            super().__init__()
            self.fc1_1 = nn.Linear(x_size, 1024)
            self.fc1_2 = nn.Linear(class_num, 1024)
            self.fc2 = nn.Linear(self.fc1_1.out_features + self.fc1_2.out_features, 512)
            self.fc2_bn = nn.BatchNorm1d(self.fc2.out_features)
            self.fc3 = nn.Linear(self.fc2.out_features, 256)
            self.fc3_bn = nn.BatchNorm1d(self.fc3.out_features)
            self.fc4 = nn.Linear(self.fc3.out_features, y_size)

        def weight_init(self, mean, std):
            for m in self._modules:
                CGAN.normal_init(self._modules[m], mean, std)

        def forward(self, x, c):
            x_1 = F.leaky_relu(self.fc1_1(x), 0.2)
            x_2 = F.leaky_relu(self.fc1_2(c), 0.2)
            x = torch.cat([x_1, x_2], 1)
            x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
            x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
            y = F.sigmoid(self.fc4(x))
            return y

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def __init__(
            self, train_loader,
            batch_size, learning_rate=2e-4,
            z_size=100, x_size=28*28, y_size=1, class_num=10):
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.G = self.Generator(z_size=z_size, x_size=x_size, class_num=class_num)
        self.D = self.Discriminator(x_size=x_size, y_size=y_size, class_num=class_num)
        self.z_size = z_size
        self.class_num = class_num

        # todo --> customizable
        self.BCE_loss = nn.BCELoss()

        # todo --> customizable
        # todo --> weight decay setting
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=learning_rate)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=learning_rate)
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

            logging.info('Training [%d:%d] D Loss %.6f, G Loss %.6f',
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
