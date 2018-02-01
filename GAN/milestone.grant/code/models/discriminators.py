import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDiscriminator(nn.Module):
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


class ConditionalDiscriminator(nn.Module):
    def __init__(self, x_size, y_size, class_num):
        super().__init__()
        self.fc1_1 = nn.Linear(x_size, 1024)
        self.fc1_2 = nn.Linear(class_num, 1024)
        self.fc2 = nn.Linear(self.fc1_1.out_features + self.fc1_2.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, y_size)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x, c):
        x_1 = F.leaky_relu(self.fc1_1(x), 0.2)
        x_2 = F.leaky_relu(self.fc1_2(c), 0.2)
        x = torch.cat([x_1, x_2], 1)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        y = F.sigmoid(self.fc4(x))
        return y


class BNDiscriminator(nn.Module):
    def __init__(self, x_size, y_size):
        super().__init__()
        self.fc1 = nn.Linear(x_size, 1024)
        self.fc1_bn = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc2_bn = nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc3_bn = nn.BatchNorm1d(self.fc3.out_features)
        self.fc4 = nn.Linear(self.fc3.out_features, y_size)

    def weight_init(self, mean, std):
        for m in self._mouldes:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.fc1_bn(self.fc1(x)), 0.2)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.leaky_relu(self.fc4_bn(self.fc4(x)), 0.2)
        y = F.tanh(self.fc4(x))
        return y


class ConditionalBNDiscriminator(nn.Module):
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
            normal_init(self._modules[m], mean, std)

    def forward(self, x, c):
        x_1 = F.leaky_relu(self.fc1_1(x), 0.2)
        x_2 = F.leaky_relu(self.fc1_2(c), 0.2)
        x = torch.cat([x_1, x_2], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        y = F.sigmoid(self.fc4(x))
        return y


class CNNDiscriminator(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z):
        z = F.leaky_relu(self.conv1(z), 0.2)
        z = F.leaky_relu(self.conv2_bn(self.conv2(z)), 0.2)
        z = F.leaky_relu(self.conv3_bn(self.conv3(z)), 0.2)
        z = F.leaky_relu(self.conv4_bn(self.conv4(z)), 0.2)
        x = F.sigmoid(self.conv5(z))
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
