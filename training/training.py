import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.IndexDataset import get_tensor, IndexDataset
from data.Interpolation import trilinear_f_interpolation, generate_RegularGridInterpolator, \
    finite_difference_trilinear_grad
from model.NeurcompModel import Neurcomp
from model.model_utils import setup_neurcomp
from visualization.OutputToVTK import tiled_net_out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training(args):
    # M: Get volume data, set up data
    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True,
                             num_workers=args['num_workers'])  # M: create dataloader from dataset to use in training, TODO: num_workers=args['num_workers']

    # volume_interpolator = generate_RegularGridInterpolator(volume)  # M: used for accessing the volume with indices

    volume = volume.to(device)

    # M: Setup model
    model = setup_neurcomp(args['compression_ratio'], dataset.n_voxels, args['n_layers'],
                           args['d_in'], args['d_out'], args['omega_0'], args['checkpoint_path'])
    model.to(device)
    model.train()

    # M: Setup loss, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])  # betas=(0.9, 0.999)
    loss_criterion = torch.nn.MSELoss().to(device)

    # M: Training loop
    voxel_seen = 0.0
    volume_passes = 0.0
    while int(volume_passes) + 1 < args['max_pass']:

        for idx, data in enumerate(data_loader):
            # M: Access data
            raw_positions, norm_positions = data

            raw_positions = raw_positions.to(device) # M: Tensor of size [batch_size, sample_size, 3]
            norm_positions = norm_positions.to(device)
            raw_positions = raw_positions.view(-1, args['d_in']) # M: Tensor of size [batch_size x sample_size, 3]
            norm_positions = norm_positions.view(-1, args['d_in'])
            norm_positions.requires_grad = True  # M: For gradient calculation of nw

            # M: NW prediction
            optimizer.zero_grad()
            predicted_volume = model(norm_positions)
            predicted_volume = predicted_volume.squeeze(-1)  # M: Tensor of size [batch_size x dataset.sample_size, 1]

            # M: Calculate loss
            ground_truth_volume = trilinear_f_interpolation(raw_positions, volume,
                                                            dataset.min_idx.to(device), dataset.max_idx.to(device),
                                                            dataset.vol_res.to(device))
            # ground_truth_volume = volume_interpolator(raw_positions.cpu())

            vol_loss = loss_criterion(predicted_volume, ground_truth_volume)
            debug_for_volumeloss = vol_loss.item()
            grad_loss = 0.0
            complete_loss = vol_loss

            if args['grad_lambda'] > 0:
                target_grad = finite_difference_trilinear_grad(raw_positions, volume,
                                                               dataset.min_idx.to(device), dataset.max_idx.to(device),
                                                               dataset.vol_res.to(device), scale=dataset.scales)
                # tg2 = numpy.gradient(volume.numpy())
                # test = tg2(raw_positions[0,0],raw_positions[0,1],raw_positions[0,2])
                # M: Important to set retain_graph, create_graph, allow_unused here, for correct gradient calculation!
                predicted_grad = torch.autograd.grad(outputs=predicted_volume, inputs=norm_positions,
                                                     grad_outputs=torch.ones_like(predicted_volume),
                                                     retain_graph=True, create_graph=True, allow_unused=False)[0]
                grad_loss = loss_criterion(target_grad, predicted_grad)
                complete_loss += args['grad_lambda'] * grad_loss

            complete_loss.backward()
            optimizer.step()

            # M: Learning rate decay
            prior_volume_passes = int(voxel_seen / dataset.n_voxels)
            voxel_seen += ground_truth_volume.shape[0]
            volume_passes = voxel_seen / dataset.n_voxels

            if prior_volume_passes != int(volume_passes) and (int(volume_passes)+1) % args['pass_decay'] == 0:
                print('------ learning rate decay ------', volume_passes)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args['lr_decay']

            # M: Print training statistics:
            if idx % 100 == 0:
                print('Passes: ', volume_passes, " volume loss: ", debug_for_volumeloss,
                      " grad loss: ", grad_loss.item(), " loss: ", complete_loss.item())

            # M: Stop training, if we reach max amount of passes over volume
            if (int(volume_passes) + 1) == args['max_pass']:
                break

    # M: print, save verbose information
    tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True, write_vols=True)

    info = {}
    num_net_params = 0
    for layer in model.parameters():
        num_net_params += layer.numel()
    compression_ratio = dataset.n_voxels/num_net_params
    print("Trained Model: ", num_net_params," parameters; ", compression_ratio, " compression ratio")
    info['volume_size'] = dataset.vol_res
    info['volume_num_voxels'] = dataset.n_voxels
    info['num_parameters'] = num_net_params
    info['compression_ratio'] = compression_ratio

    print("Layers: \n", model)

    ExperimentPath = os.path.abspath(os.getcwd()) + args['basedir'] + args['expname'] + '/'
    os.makedirs(ExperimentPath, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(ExperimentPath,'model.pth'))

    def write_dict(dictionary, filename):
        with open(os.path.join(ExperimentPath, filename), 'w') as f:
            for key, value in dictionary.items():
                f.write('%s = %s\n' % (key, value))

    write_dict(args, 'config.txt')
    write_dict(info, 'info.txt')

