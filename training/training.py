import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.IndexDataset import get_tensor_from_numpy, IndexDataset
from data.Interpolation import trilinear_f_interpolation, generate_RegularGridInterpolator, finite_difference_trilinear_grad
from model.NeurcompModel import Neurcomp, compute_num_neurons

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUMPYFILE = 'datasets/test_vol.npy'  # M: SIze: 150^3


def training(args):
    # M: get volume data, set up data
    volume = get_tensor_from_numpy(NUMPYFILE)
    dataset = IndexDataset(volume, 16)
    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True,
                             )  # M: create dataloader from dataset to use in training, TODO: num_workers=8

    #volume_interpolator = generate_RegularGridInterpolator(volume)  # M: used for accessing the volume with indices

    volume = volume.to(device)
    dataset.move_data_to_device(device)

    n_layers = 4    # M: TODO parse this in
    compression_ratio = 50  # M: TODO parse this in
    max_epochs = 100
    learning_rate = 5e-5

    # M: setup model
    target_size = int(dataset.n_voxels / compression_ratio) # M: Amt of neurons in whole model
    num_neurons = compute_num_neurons(num_layer=n_layers, target_size=target_size)  # M: number of neurons per layer
    feature_list = np.full(n_layers, num_neurons) # M: list holding the amt of neurons per layer

    model = Neurcomp(input_ch=3, output_ch=1, features=feature_list)
    model.to(device)
    model.train()

    # M: setup loss, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #betas=(0.9, 0.999)
    loss_criterion = torch.nn.MSELoss().to(device)

    # M: print verbose information


    # M: training loop
    n_iter = 0
    for epoch in range(max_epochs):

        for bdx, data in enumerate(data_loader):
            n_iter += 1

            # M: access data
            raw_positions, norm_positions = data
            raw_positions = raw_positions.to(device)
            norm_positions = norm_positions.to(device)
            raw_positions = raw_positions.view(-1, 3)
            norm_positions = norm_positions.view(-1, 3)

            # M: NW prediction
            optimizer.zero_grad()
            predicted_volume = model(norm_positions)
            predicted_volume = predicted_volume.squeeze(-1) # M: Tensor holding batch_size x dataset.sample_size entries

            # M: Calculate loss
            ground_truth_volume = trilinear_f_interpolation(raw_positions, volume,
                                                            dataset.min_idx, dataset.max_idx, dataset.vol_res)
            #ground_truth_volume = volume_interpolator(raw_positions.cpu())

            target_grad = finite_difference_trilinear_grad(raw_positions, volume, dataset.min_idx, dataset.max_idx,
                                                           dataset.vol_res, scale=dataset.scales)

            #tg2 = numpy.gradient(volume.numpy())
            #test = tg2(raw_positions[0,0],raw_positions[0,1],raw_positions[0,2])


            print("yep", n_iter)



if __name__ == '__main__':
    args = {}
    training(args)
