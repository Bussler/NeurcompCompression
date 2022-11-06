import os

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.IndexDataset import get_tensor, IndexDataset
from data.Interpolation import trilinear_f_interpolation, finite_difference_trilinear_grad
from model.NeurcompModel import Neurcomp
from model.model_utils import setup_neurcomp
from visualization.OutputToVTK import tiled_net_out
from mlflow import log_metric, log_param, log_artifacts
from model.SmallifyDropoutLayer import calculte_smallify_loss, SmallifyDropout, sign_variance_pruning_strategy,\
    SmallifyResidualSiren, sign_variance_pruning_strategy_OD
from model.pruning import prune_dropout_threshold, prune_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_Mlfow_Experiment(name):
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    found_experiment = None
    for experiment in client.search_experiments():
        if experiment.name == name:
            found_experiment = experiment
            break
    if found_experiment == None:
        experiment_id = mlflow.create_experiment(name)
    else:
        experiment_id = found_experiment.experiment_id
    return experiment_id


def gather_training_info(model, dataset, volume, args, verbose=True):
    psnr, l1_diff, mse, rmse = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True,
                                                 write_vols=True)

    # M: print, save verbose information
    info = {}
    num_net_params = 0
    for layer in model.parameters():
        num_net_params += layer.numel()
    compression_ratio = dataset.n_voxels / num_net_params

    if verbose:
        print("Trained Model: ", num_net_params, " parameters; ", compression_ratio, " compression ratio")

    info['volume_size'] = dataset.vol_res
    info['volume_num_voxels'] = dataset.n_voxels
    info['num_parameters'] = num_net_params
    info['network_layer_sizes'] = model.layer_sizes
    info['compression_ratio'] = compression_ratio
    info['psnr'] = psnr
    info['l1_diff'] = l1_diff
    info['mse'] = mse
    info['rmse'] = rmse

    log_param("num_net_params", num_net_params)
    log_param("compression_ratio", compression_ratio)
    log_param("volume_size", dataset.vol_res)
    log_param("volume_num_voxels", dataset.n_voxels)
    log_param("network_layer_sizes", model.layer_sizes)
    log_param("psnr", psnr)
    log_param("l1_diff", l1_diff)
    log_param("mse", mse)
    log_param("rmse", rmse)

    if args['dropout_technique']:
        log_param("lambda_Betas", args['lambda_betas'])
        log_param("lambda_Weights", args['lambda_weights'])

    if verbose:
        print("Layers: ", model.layer_sizes)
        print("Model: \n", model)

    ExperimentPath = os.path.abspath(os.getcwd()) + args['basedir'] + args['expname'] + '/'
    os.makedirs(ExperimentPath, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(ExperimentPath, 'model.pth'))
    args['checkpoint_path'] = os.path.join(ExperimentPath, 'model.pth')
    args['feature_list'] = model.layer_sizes[1:-1]

    def write_dict(dictionary, filename):
        with open(os.path.join(ExperimentPath, filename), 'w') as f:
            for key, value in dictionary.items():
                f.write('%s = %s\n' % (key, value))

    write_dict(args, 'config.txt')
    write_dict(info, 'info.txt')

    log_artifacts(ExperimentPath)

    return info


def training(args, verbose=True):
    # M: Get volume data, set up data
    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True,
                             num_workers=args['num_workers'])  # M: create dataloader from dataset to use in training
    volume = volume.to(device)

    # M: Setup model
    model = setup_neurcomp(args['compression_ratio'], dataset.n_voxels, args['n_layers'], args['d_in'],
                           args['d_out'], args['omega_0'], args['checkpoint_path'], args['dropout_technique'],
                           args['pruning_momentum'])
    model.to(device)
    model.train()

    # M: Setup loss, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_criterion = torch.nn.MSELoss().to(device)
    #loss_criterion = torch.nn.L1Loss().to(device)  # M: try L1 loss

    # M: Training loop
    voxel_seen = 0.0
    volume_passes = 0.0
    step_iter = 0

    last_loss = None
    no_gain_iter = 0

    mlflow.start_run(experiment_id=get_Mlfow_Experiment(args['expname']))

    while int(volume_passes) + 1 < args['max_pass']:  # M: epochs

        for idx, data in enumerate(data_loader):
            step_iter += 1

            # M: Access data
            raw_positions, norm_positions = data

            raw_positions = raw_positions.to(device)  # M: Tensor of size [batch_size, sample_size, 3]
            norm_positions = norm_positions.to(device)
            raw_positions = raw_positions.view(-1, args['d_in'])  # M: Tensor of size [batch_size x sample_size, 3]
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

            vol_loss = loss_criterion(predicted_volume, ground_truth_volume)
            debug_for_volumeloss = vol_loss.item()
            grad_loss = 0.0
            complete_loss = vol_loss

            if args['grad_lambda'] > 0:  # M: Gradient loss
                target_grad = finite_difference_trilinear_grad(raw_positions, volume,
                                                               dataset.min_idx.to(device), dataset.max_idx.to(device),
                                                               dataset.vol_res.to(device), scale=dataset.scales)

                # M: Important to set retain_graph, create_graph, allow_unused here, for correct gradient calculation!
                predicted_grad = torch.autograd.grad(outputs=predicted_volume, inputs=norm_positions,
                                                     grad_outputs=torch.ones_like(predicted_volume),
                                                     retain_graph=True, create_graph=True, allow_unused=False)[0]
                grad_loss = loss_criterion(predicted_grad, target_grad)
                complete_loss += args['grad_lambda'] * grad_loss

            # M: Special loss for Dropout
            if args['dropout_technique']:
                if args['dropout_technique'] == 'smallify':
                    loss_Betas, loss_Weights = calculte_smallify_loss(model)
                    complete_loss = complete_loss + (loss_Betas * args['lambda_betas'])\
                                    + (loss_Weights * args['lambda_weights'])

                    #prune_dropout_threshold(model, SmallifyDropout, threshold=0.1)
                    pruned = sign_variance_pruning_strategy(model, optimizer, device, threshold=args['pruning_threshold'])
                    #sign_variance_pruning_strategy_OD(model, device, threshold=args['pruning_threshold'])

                    if pruned:
                        lr_list = []
                        print("--CHANGING OPTIM--")
                        for param_group in optimizer.param_groups:
                            lr_list.append(param_group['lr'])
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr_list[0])
                        for index, param_group in enumerate(optimizer.param_groups):
                            param_group['lr'] = lr_list[index]

            complete_loss.backward()
            optimizer.step()

            # M: Learning rate decay
            prior_volume_passes = int(voxel_seen / dataset.n_voxels)
            voxel_seen += ground_truth_volume.shape[0]
            volume_passes = voxel_seen / dataset.n_voxels

            if prior_volume_passes != int(volume_passes) and (int(volume_passes) + 1) % args['pass_decay'] == 0:
                print('------ learning rate decay ------', volume_passes)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args['lr_decay']

            # M: Print training statistics:
            if idx % 100 == 0 and verbose:
                if args['grad_lambda'] > 0:
                    print('Pass [{:.4f} / {:.1f}]: volume loss: {:.4f}, grad loss: {:.4f}, mse: {:.4f}'.format(
                        volume_passes, args['max_pass'], debug_for_volumeloss, grad_loss.item(), complete_loss.item()))
                else:
                    print('Pass [{:.4f} / {:.1f}]: volume loss: {:.4f}, mse: {:.4f}'.format(
                        volume_passes, args['max_pass'], debug_for_volumeloss, complete_loss.item()))
                if args['dropout_technique']:
                    print('Beta Loss: {:.4f}, Weight loss: {:.4f}'.format(loss_Betas, loss_Weights))

            log_metric(key="loss", value=complete_loss.item(), step=step_iter)
            log_metric(key="volume_loss", value=debug_for_volumeloss, step=step_iter)

            if args['grad_lambda'] > 0:
                log_metric(key="grad_loss", value=grad_loss.item(), step=step_iter)
            if args['dropout_technique']:
                log_metric(key="beta_loss", value=loss_Betas, step=step_iter)
                log_metric(key="nw_weight_loss", value=loss_Weights, step=step_iter)

                #if last_loss is None or complete_loss < last_loss:  # M: complete loss or just beta_loss?
                #    last_loss = complete_loss
                #    no_gain_iter = 0
                #else:
                #    no_gain_iter += 1
                #if no_gain_iter == 5 and args['lambda_betas'] > 1e-7:
                #    args['lambda_betas'] /= 10.0
                #    no_gain_iter = 0

            # M: Stop training, if we reach max amount of passes over volume
            if (int(volume_passes) + 1) == args['max_pass']:
                break

    # M: remove dropout layers from model
    if args['dropout_technique']:
        intermed_layers = [model.d_in, model.layer_sizes[1]]
        if args['dropout_technique'] == 'smallify':
            #model = prune_model(model, SmallifyDropout)
            #model.to(device)
            for module in model.net_layers.modules():
                if isinstance(module, SmallifyResidualSiren):
                    c = module.remove_dropout_layers()
                    intermed_layers.append(c)

        intermed_layers.append(model.d_out)
        model.layer_sizes = intermed_layers

    info = gather_training_info(model, dataset, volume, args, verbose)
    mlflow.end_run()

    return info

