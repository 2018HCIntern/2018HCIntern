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
