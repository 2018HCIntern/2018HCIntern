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