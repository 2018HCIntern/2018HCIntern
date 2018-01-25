from torch.utils.data import Dataset, DataLoader
import pandas as pd
import click
import logging
from models.gan.model import GANDict


@click.command()
@click.option('--gan', default='GAN', help='Input prefer GAN Model')
@click.option('--size', help='Input the number of Generated Data')
@click.option('--ratio', help='Input the ratio of Generated Data per Real Data')
@click.option('--dataframe', help='Input csv file location')
@click.option('--dataset', help='Input pytorch dataset location')
@click.option('--y', default='y', help='Input the column name of y label')
@click.option('--z-size', default=100, help='Input the size of random distribution')
@click.option('--batch-size', default=128, help='Input mini batch size')
def augmentation(gan, size, ratio, dataframe, dataset, y, z_size, batch_size):
    check_arguments(size, ratio, dataframe, dataset, y)
    train_loader = load_data(dataframe, dataset, y, batch_size)
    x_size = train_loader.dataset.x_size
    class_num = train_loader.dataset.class_num
    GAN = GANDict[gan](train_loader, batch_size=batch_size, z_size=z_size, x_size=x_size, class_num=class_num)
    GAN.train(10)
    if ratio is not None:
        size = ratio * len(train_loader.dataset)
    gen_data = GAN.generate(size)


# todo --> separate this dataset class to other file
class DataframeDataset(Dataset):
    def __init__(self, dataframe, y_attr):
        self.df = pd.read_csv(dataframe)
        self.y_attr = y_attr
        self.x_size = len(self.df.columns.values) - 1
        self.class_num = len(pd.unique(self.df[self.y_attr]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.loc[idx]
        data = item.drop(self.y_attr).as_matrix()
        # only number available
        labels = item[self.y_attr]
        sample = {'data': data, 'labels': labels}
        return sample


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

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
