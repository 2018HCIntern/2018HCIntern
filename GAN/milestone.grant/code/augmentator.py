from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import click
import logging
from models.gan.model import GANDict


@click.command()
@click.option('--gan', default='GAN', help='Input prefer GAN Model')
@click.option('--size', type=int, help='Input the number of Generated Data')
@click.option('--ratio', type=float, help='Input the ratio of Generated Data per Real Data')
@click.option('--dataframe', help='Input csv file location')
@click.option('--dataset', help='Input pytorch dataset location')
# @click.option('--save-path', help='Input Generated + Original DataFrame Save Path')
@click.option('--y', default='y', help='Input the column name of y label')
@click.option('--z-size', default=100, help='Input the size of random distribution')
@click.option('--batch-size', default=128, help='Input mini batch size')
def augmentation(gan, size, ratio, dataframe, dataset, y, z_size, batch_size):
    check_arguments(size, ratio, dataframe, dataset, y)
    df = pd.read_csv(dataframe)
    df_mean = df.mean()
    df_std = df.std()
    df = (df - df_mean) / df_std
    train_loader = load_data(df, dataset, y, batch_size)
    x_size = train_loader.dataset.x_size
    class_num = train_loader.dataset.class_num
    GAN = GANDict[gan](train_loader, batch_size=batch_size, z_size=z_size, x_size=x_size, class_num=class_num)
    GAN.train(100)
    if ratio is not None:
        size = ratio * len(train_loader.dataset)
    gen_data = GAN.generate(size)
    gen_data = gen_data * df_std + df_mean
    print(gen_data)
    return GAN, pd.concat([df, gen_data])


# todo --> separate this dataset class to other file
class DataframeDataset(Dataset):
    def __init__(self, dataframe, y_attr):
        self.df = dataframe
        self.y_attr = y_attr
        self.x_size = len(self.df.columns.values) - 1
        self.class_num = len(pd.unique(self.df[self.y_attr]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.loc[idx]
        data = item.drop(self.y_attr).as_matrix()
        labels = item[self.y_attr]
        data = data.astype('float32')
        labels = labels.astype('float32')
        return (data, labels)


def check_arguments(size, ratio, dataframe, dataset, y):
    if (dataframe is None) == (dataset is None):
        logging.error('Data Input Error')
        exit()

    if (size is None) == (ratio is None):
        logging.error('Size Input Error')
        exit()


def load_data(dataframe, dataset, y, batch_size):
    if dataframe is not None:
        dataset = DataframeDataset(dataframe, y)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader
