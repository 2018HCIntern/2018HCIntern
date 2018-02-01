import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGenerator(nn.Module):
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


class ConditionalGenerator(nn.Module):
    def __init__(self, z_size, x_size, class_num):
        super().__init__()
        self.fc1_1 = nn.Linear(z_size, 256)
        self.fc1_2 = nn.Linear(class_num, 256)
        self.fc2 = nn.Linear(self.fc1_1.out_features + self.fc1_2.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, x_size)

    def forward(self, z, c):
        z_1 = F.leaky_relu(self.fc1_1(z), 0.2)
        z_2 = F.leaky_relu(self.fc1_2(c), 0.2)
        z = torch.cat([z_1, z_2], 1)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        x = F.tanh(self.fc4(z))
        return x


class BNGenerator(nn.Module):
    def __init__(self, z_size, x_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, 256)
        self.fc1_bn = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc2_bn = nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc3_bn = nn.BatchNorm1d(self.fc3.out_features)
        self.fc4 = nn.Linear(self.fc3.out_features, x_size)

    def weight_init(self, mean, std):
        for m in self._mouldes:
            normal_init(self._modules[m], mean, std)

    def forward(self, z):
        z = F.relu(self.fc1_bn(self.fc1(z)))
        z = F.relu(self.fc2_bn(self.fc2(z)))
        z = F.relu(self.fc3_bn(self.fc3(z)))
        z = F.relu(self.fc4_bn(self.fc4(z)))
        x = F.tanh(self.fc4(z))
        return x


class ConditionalBNGenerator(nn.Module):
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
            normal_init(self._modules[m], mean, std)

    def forward(self, z, c):
        z_1 = F.relu(self.fc1_1_bn(self.fc1_1(z)))
        z_2 = F.relu(self.fc1_2_bn(self.fc1_2(c)))
        z = torch.cat([z_1, z_2], 1)
        z = F.relu(self.fc2_bn(self.fc2(z)))
        z = F.relu(self.fc3_bn(self.fc3(z)))
        x = F.tanh(self.fc4(z))
        return x


class CNNGenerator(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z):
        z = F.relu(self.deconv1_bn(self.deconv1(z)))
        z = F.relu(self.deconv2_bn(self.deconv2(z)))
        z = F.relu(self.deconv3_bn(self.deconv3(z)))
        z = F.relu(self.deconv4_bn(self.deconv4(z)))
        x = F.tanh(self.deconv5(z))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
