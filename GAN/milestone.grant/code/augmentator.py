from __future__ import print_function
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.models import model

parser = argparse.ArgumentParser()
parser.add_argument('--gan_type', default='GAN', help=' GAN | CGAN | WGAN | WCGAN | VIGAN ')
parser.add_argument('--gen_num', type=int, default=64)
parser.add_argument('--data_root', required=True)
parser.add_argument('--data_type', help=' FRAME | SET ')
parser.add_argument('--y_label', default=None, help='input Y label')
parser.add_argument('--z_dim', type=int, default=100, help='input Z dimension')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch_num', type=int, default=100, help='input epoch number')
parser.add_argument('--lrD', type=float, default=0.00005)
parser.add_argument('--lrG', type=float, default=0.00005)
parser.add_argument('--input_norm', action='store_true')
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--image', action='store_true')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--path', type=str, default='./result')
opt = parser.parse_args()

class DataFrameDataset(Dataset):
    def __init__(self, dataFrame, y_label):
        self.df = pd.read_csv(dataFrame)
        self.y_label = y_label
        self.x_dim = len(self.df.columns.values) - 1
        self.class_num = len(pd.unique(self.df[self.y_label]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.loc[index]
        data = item.drop(self.y_label).as_matrix().astype('float32')
        labels = item[self.y_label].astype('long')
        return (data, labels)

def main():
    data_loader = load_data()
    z_dim, x_dim, class_num = opt.z_dim, data_loader.dataset.x_dim, data_loader.dataset.class_num
    GAN = model[opt.gan_type](
        train_loader=data_loader,
        batch_size=opt.batch_size,
        x_size=x_dim,
        z_size=z_dim,
        lrG=opt.lrG,
        lrD=opt.lrD,
        epoch_num=opt.epoch_num,
        class_num=class_num,
        clamp_lower=opt.clamp_lower,
        clamp_upper=opt.clamp_upper)
    GAN.train(opt.epoch_num)
    gen_data = GAN.generate(opt.gen_num)
    GAN.save()
    return GAN, gen_data

def load_data():
    if opt.data_type == 'FRAME':
        dataset = DataFrameDataset(opt.data_root, opt.y_label)
    elif opt.data_type == 'SET':
        dataset = dset.ImageFolder(root=opt.data_root,
                                   transform=transforms.Compose([
                                       transforms.Scale(opt.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    else:
        # error handling
        dataset = None
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

if __name__ == '__main__':
    main()