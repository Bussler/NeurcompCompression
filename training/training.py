import numpy as np
import torch
from torch.utils.data import DataLoader

from data.IndexDataset import get_tensor_from_numpy, IndexDataset
from data.testDataset import VolumeDataset

NUMPYFILE = 'datasets/test_vol.npy' # M: SIze: 150^3


def training(args):
    volume = get_tensor_from_numpy(NUMPYFILE)

    dataset = IndexDataset(volume, 16)

    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True,
                             )  # M: create dataloader from dataset to use in training, num_workers=8

    n_iter = 0
    for bdx, data in enumerate(data_loader):
        n_iter += 1
        raw_positions, positions = data
        print("yep", n_iter)


if __name__=='__main__':
    args = {}
    training(args)