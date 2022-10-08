import numpy as np
import torch
from torch.utils.data import DataLoader

from data.IndexDataset import get_tensor_from_numpy, IndexDataset
from model.NeurcompModel import Neurcomp, compute_num_neurons

NUMPYFILE = 'datasets/test_vol.npy'  # M: SIze: 150^3


def training(args):
    volume = get_tensor_from_numpy(NUMPYFILE)

    dataset = IndexDataset(volume, 16)

    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True,
                             )  # M: create dataloader from dataset to use in training, TODO: num_workers=8

    n_layers = 4    # M: TODO parse this in
    compression_ratio = 50  # M: TODO parse this in

    target_size = int(dataset.n_voxels / compression_ratio)
    num_neurons = compute_num_neurons(num_layer=n_layers, target_size=target_size)  # M: number of neurons per layer
    feature_list = np.full(n_layers, num_neurons)

    model = Neurcomp(input_ch=3, output_ch=1, features=feature_list)

    n_iter = 0
    # for bdx, data in enumerate(data_loader):
    #    n_iter += 1
    #    raw_positions, positions = data
    #    print("yep", n_iter)


if __name__ == '__main__':
    args = {}
    training(args)
