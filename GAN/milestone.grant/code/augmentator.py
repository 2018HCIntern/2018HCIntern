from __future__ import print_function
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from models.models import model


class DataFrameDataset(Dataset):
    def __init__(self, dataFrame, y_label):
        self.df = pd.read_csv(dataFrame)
        self.y_label = y_label
        self.x_dim = len(self.df.columns.values) - 1
        self.class_num = len(pd.unique(self.df[self.y_label]))
        self.categories = { }
        self.dataCheck()
        self.dfmax = self.df.max()
        self.dfmin = self.df.min()
        # print(self.categories)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.loc[index]
        data = item.drop(self.y_label).as_matrix().astype('float32')
        labels = item[self.y_label].astype('long')
        return (data, labels)

    def dataCheck(self):
        threshold = len(self) // 20 # is it good ratio?
        aY = False
        for col in self.df.columns:
            unique = np.sort(self.df[col].unique())

            if self.df[col].dtype == np.int or len(unique) < threshold:
                if aY:
                    self.categories[col] = unique
                    continue
                from tqdm import tqdm
                tqdm.write('\n'.join([
                               'Expecting the column {} is categorical or one hot value'.format(col),
                               'If you want to generated data be in those categories, type [Y]/n/aY/aN'
                           ]))
                ans = input()
                if ans.lower() == 'y' or ans == '':
                    self.categories[col] = unique
                elif ans.lower() == 'ay':
                    self.categories[col] = unique
                    aY = True
                elif ans.lower() == 'an':
                    break

    def dataNorm(self):
        # print(self.df['A..papers'])
        self.df = (self.df - self.dfmin) / (self.dfmax - self.dfmin)
        # print(self.df['A..papers'])

    def dataDenorm(self, df=None):
        if df is None:
            self.df = self.df * (self.dfmax - self.dfmin) + self.dfmin
        else:
            # print(df['A..papers'])
            df = df * (self.dfmax - self.dfmin) + self.dfmin
            # print(df['A..papers'])

    def dataRound(self, df):
        from bisect import bisect_left
        from tqdm import tqdm
        pbar = tqdm(total = self.categories)
        def round_val(mlist, mnum):
            pos = bisect_left(mlist, mnum)
            if pos == 0:
                return mlist[0]
            if pos == len(mlist):
                return mlist[-1]
            before = mlist[pos - 1]
            after = mlist[pos]
            if after - mnum < mnum - before:
                return after
            else:
                return before

        for col in self.categories:
            rounder = self.categories[col]
            for i in range(len(df)):
                df[col][i] = round_val(rounder, df[col][i])
            pbar.update(1)



def main():
    # testing
    if opt.image:
        data_loader = torch.utils.data.DataLoader(
            dset.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Lambda(lambda x: x.view(-1))
                           ])),
            batch_size=opt.batch_size, shuffle=True)
    # testing
    else:
        data_loader = load_data()

    # testing
    if opt.image:
        z_dim, x_dim, class_num = 100, 784, 10
    else:
        z_dim, x_dim, class_num = opt.z_dim, data_loader.dataset.x_dim, data_loader.dataset.class_num
    GAN = model[opt.gan_type](
        train_loader=data_loader,
        batch_size=opt.batch_size,
        x_size=x_dim,
        z_size=z_dim,
        y_size=1,
        lrG=opt.lrG,
        lrD=opt.lrD,
        epoch_num=opt.epoch_num,
        class_num=class_num,
        clamp_lower=opt.clamp_lower,
        clamp_upper=opt.clamp_upper,
        image=opt.image)
    GAN.train(opt.epoch_num)
    gen_data = GAN.generate(opt.gen_num)
    data_loader.dataset.dataDenorm(gen_data)
    data_loader.dataset.dataRound(gen_data)
    GAN.save('{}/generator_weight'.format(opt.path), '{}/discriminator_weight'.format(opt.path))
    gen_data.to_csv('{}/gen_data.csv'.format(opt.path), index=False)
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
    dataset.dataNorm()
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

if __name__ == '__main__':
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

    main()