import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.IndexDataset import get_tensor_from_numpy, IndexDataset
from data.Interpolation import trilinear_f_interpolation, generate_RegularGridInterpolator, \
    finite_difference_trilinear_grad
from model.NeurcompModel import Neurcomp, compute_num_neurons

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUMPYFILE = 'datasets/test_vol.npy'  # M: SIze: 150^3


def training(args):
    # M: get volume data, set up data
    volume = get_tensor_from_numpy(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True,
                             )  # M: create dataloader from dataset to use in training, TODO: num_workers=args['num_workers']

    # volume_interpolator = generate_RegularGridInterpolator(volume)  # M: used for accessing the volume with indices

    volume = volume.to(device)
    dataset.move_data_to_device(device)

    # M: setup model
    target_size = int(dataset.n_voxels / args['compression_ratio'])  # M: Amt of neurons in whole model
    num_neurons = compute_num_neurons(num_layer=args['n_layers'],
                                      target_size=target_size)  # M: number of neurons per layer
    feature_list = np.full(args['n_layers'], num_neurons)  # M: list holding the amt of neurons per layer

    model = Neurcomp(input_ch=args['d_in'], output_ch=args['d_out'], features=feature_list)
    model.to(device)
    model.train()

    # M: setup loss, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])  # betas=(0.9, 0.999)
    loss_criterion = torch.nn.MSELoss().to(device)

    # M: training loop
    pass_iter = 0
    for epoch in range(args['max_epochs']):

        for idx, data in enumerate(data_loader):
            pass_iter += 1

            # M: access data
            raw_positions, norm_positions = data

            raw_positions = raw_positions.to(device)
            norm_positions = norm_positions.to(device)
            raw_positions = raw_positions.view(-1, args['d_in'])
            norm_positions = norm_positions.view(-1, args['d_in'])
            norm_positions.requires_grad = True  # M: For gradient calculation of nw

            # M: NW prediction
            optimizer.zero_grad()
            predicted_volume = model(norm_positions)
            predicted_volume = predicted_volume.squeeze(-1)  # M: Tensor holding batch_size x dataset.sample_size entries

            # M: Calculate loss
            ground_truth_volume = trilinear_f_interpolation(raw_positions, volume,
                                                            dataset.min_idx, dataset.max_idx, dataset.vol_res)
            # ground_truth_volume = volume_interpolator(raw_positions.cpu())

            vol_loss = loss_criterion(predicted_volume, ground_truth_volume)
            complete_loss = vol_loss

            if args['grad_lambda'] > 0:
                target_grad = finite_difference_trilinear_grad(raw_positions, volume, dataset.min_idx, dataset.max_idx,
                                                               dataset.vol_res, scale=dataset.scales)
                # tg2 = numpy.gradient(volume.numpy())
                # test = tg2(raw_positions[0,0],raw_positions[0,1],raw_positions[0,2])
                predicted_grad = torch.autograd.grad(outputs=predicted_volume, inputs=norm_positions,
                                                     grad_outputs=torch.ones_like(predicted_volume),
                                                     retain_graph=True)[0]
                grad_loss = loss_criterion(target_grad, predicted_grad)
                complete_loss += args['grad_lambda'] * grad_loss

            complete_loss.backward()
            optimizer.step()

            # M: learning rate decay
            if(pass_iter+1) % args['pass_decay'] == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args['lr_decay']

            print("yep", pass_iter)

    # M: print verbose information
    pass


if __name__ == '__main__':
    args = {}
    training(args)
