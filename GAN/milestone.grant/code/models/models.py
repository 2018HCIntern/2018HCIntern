from models.gan import GAN
from .cgan import CGAN
# from .wgan import WGAN
# from .wcgan import WCGAN
# from .dagan import DAGAN

model = {
    'GAN': GAN,
    'CGAN': CGAN,
    # 'WGAN': WGAN,
    # 'WCGAN': WCGAN,
    # 'DAGAN': DAGAN
}

class GANConfig(object):
    def __init__(self, **kwargs):
        self.train_loader = None
        self.batch_size = None
        self.x_size = None
        self.z_size = None
        self.lrG = None
        self.lrD = None
        self.epoch_num = None
        self.class_num = None
        self.save_path = None
        self.clamp_lower = None
        self.clamp_upper = None
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
