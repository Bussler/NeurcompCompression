import os

import mlflow
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader

from data.IndexDataset import get_tensor, IndexDataset
from data.Interpolation import trilinear_f_interpolation, finite_difference_trilinear_grad
from model.NeurcompModel import Neurcomp
from model.model_utils import setup_neurcomp, write_dict
from visualization.OutputToVTK import tiled_net_out
from mlflow import log_metric, log_param, log_artifacts
from model.SmallifyDropoutLayer import calculte_smallify_loss, SmallifyDropout, sign_variance_pruning_strategy_dynamic,\
    SmallifyResidualSiren, sign_variance_pruning_strategy_do_prune
from model.pruning import prune_dropout_threshold, prune_smallify_use_resnet, prune_smallify_no_Resnet,\
    prune_variational_dropout_no_resnet, prune_variational_dropout_use_resnet
from model.VariationalDropoutLayer import calculate_variational_dropout_loss, inference_variational_model, VariationalDropout
import training.learning_rate_decay as lrdecay
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = None


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
    is_probalistic_model = 'variational' in args['dropout_technique']
    psnr, l1_diff, mse, rmse = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True,
                                             probalistic_model=is_probalistic_model, write_vols=True)

    # M: print, save verbose information
    info = {}
    num_net_params = 0
    for layer in model.parameters():
        num_net_params += layer.numel()
    compression_ratio = dataset.n_voxels / num_net_params

    if verbose:
        print("Trained Model: ", num_net_params, " parameters; ", compression_ratio, " compression ratio")

    info['volume_size'] = dataset.vol_res.tolist()
    info['volume_num_voxels'] = dataset.n_voxels
    info['num_parameters'] = num_net_params
    info['network_layer_sizes'] = model.layer_sizes
    info['compression_ratio'] = compression_ratio
    info['psnr'] = psnr
    info['l1_diff'] = l1_diff
    info['mse'] = mse
    info['rmse'] = rmse

    #log_param("num_net_params", num_net_params)
    #log_param("compression_ratio", compression_ratio)
    #log_param("volume_size", dataset.vol_res)
    #log_param("volume_num_voxels", dataset.n_voxels)
    #log_param("network_layer_sizes", model.layer_sizes)
    #log_param("psnr", psnr)
    #log_param("l1_diff", l1_diff)
    #log_param("mse", mse)
    #log_param("rmse", rmse)
    writer.add_scalar("compression_ratio", compression_ratio)
    writer.add_scalar("psnr", psnr)
    writer.add_scalar("mse", mse)
    writer.add_scalar("rmse", rmse)

    if args['dropout_technique']:
        writer.add_scalar("lambda_Betas", args['lambda_betas'])
        writer.add_scalar("lambda_Weights", args['lambda_weights'])
        writer.add_scalar("variational_init_droprate", args['variational_init_droprate'])
        writer.add_scalar("pruning_threshold", args['pruning_threshold'])

    if verbose:
        print("Layers: ", model.layer_sizes)
        print("Model: \n", model)

    ExperimentPath = os.path.abspath(os.getcwd()) + args['basedir'] + args['expname'] + '/'
    os.makedirs(ExperimentPath, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(ExperimentPath, 'model.pth'))
    args['checkpoint_path'] = os.path.join(ExperimentPath, 'model.pth')
    args['feature_list'] = model.layer_sizes[1:-1]

    write_dict(args, 'config.txt', ExperimentPath)
    write_dict(info, 'info.txt', ExperimentPath)

    #log_artifacts(ExperimentPath)

    return info


def solveModel(model_init, optimizer, lrStrategy, loss_criterion, volume, dataset, data_loader, args, verbose):
    # M: Training loop
    model = model_init
    voxel_seen = 0.0
    volume_passes = 0.0
    step_iter = 0
    lr_decay_stop = False

    while int(volume_passes) + 1 < args['max_pass'] and not lr_decay_stop:  # M: epochs

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

            # M: Used for Learning rate decay
            prior_volume_passes = int(voxel_seen / dataset.n_voxels)
            voxel_seen += ground_truth_volume.shape[0]
            volume_passes = voxel_seen / dataset.n_voxels

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
                complete_loss = complete_loss + (args['grad_lambda'] * grad_loss)

            # M: Special loss for Dropout
            if args['dropout_technique']:
                if args['dropout_technique'] == 'smallify':
                    loss_Betas, loss_Weights = calculte_smallify_loss(model)
                    complete_loss = complete_loss + (loss_Betas * args['lambda_betas']) \
                                    + (loss_Weights * args['lambda_weights'])
                if args['dropout_technique'] == 'variational':
                    complete_loss, dkl, ll, mse, loss_Weights = calculate_variational_dropout_loss(model, loss_criterion,
                                                                         predicted_volume, ground_truth_volume,
                                                                         log_sigma=args['variational_sigma'],
                                                                         pass_init_losses=args['variational_dkl_scalinginit'],
                                                                         current_pass=0, lambda_dkl=args['variational_lambda_dkl'],
                                                                         lambda_weights=args['variational_lambda_weight'],
                                                                         lambda_entropy=args['variational_lambda_entropy'])

            complete_loss.backward()
            optimizer.step()

            # M: Learning rate decay
            if lrStrategy.decay_learning_rate(prior_volume_passes, volume_passes, complete_loss):
                lr_decay_stop = True
                break

            # M: Print training statistics:
            if idx % 100 == 0 and verbose:
                if args['dropout_technique'] and args['dropout_technique'] == 'variational':
                    print(
                        'Pass [{:.4f} / {:.1f}]: volume mse: {:.4f}, LL: {:.4f}, DKL: {:.4f}, complete loss: {:.4f} '
                        .format(volume_passes, args['max_pass'], mse.item(), ll, dkl, complete_loss.item()))

                    valid_fraction = []
                    bin1 = []
                    bin2 = []
                    bin3= []
                    for module in model.net_layers.modules():
                        if isinstance(module, VariationalDropout):
                            d, b1, b2, b3 = module.get_valid_fraction()
                            valid_fraction.append(d)
                            bin1.append(b1)
                            bin2.append(b2)
                            bin3.append(b3)
                    writer.add_scalar("bin1_layer1", bin1[0], step_iter)
                    writer.add_scalar("bin1_layer2", bin1[1], step_iter)
                    writer.add_scalar("bin1_layer3", bin1[2], step_iter)
                    writer.add_scalar("bin2_layer1", bin2[0], step_iter)
                    writer.add_scalar("bin2_layer2", bin2[1], step_iter)
                    writer.add_scalar("bin2_layer3", bin2[2], step_iter)
                    writer.add_scalar("bin3_layer1", bin3[0], step_iter)
                    writer.add_scalar("bin3_layer2", bin3[1], step_iter)
                    writer.add_scalar("bin3_layer3", bin3[2], step_iter)
                    print('Valid Fraction: ', valid_fraction)
                else:
                    if args['grad_lambda'] > 0:
                        print('Pass [{:.4f} / {:.1f}]: volume loss: {:.4f}, grad loss: {:.4f}, mse: {:.6f}'.format(
                            volume_passes, args['max_pass'], debug_for_volumeloss, grad_loss.item(), complete_loss.item()))
                    else:
                        print('Pass [{:.4f} / {:.1f}]: volume loss: {:.4f}, mse: {:.6f}'.format(
                            volume_passes, args['max_pass'], debug_for_volumeloss, complete_loss.item()))
                    if args['dropout_technique']:
                        if args['dropout_technique'] == 'smallify':
                            print('Beta Loss: {:.4f}, Weight loss: {:.4f}'.format(loss_Betas, loss_Weights))

            writer.add_scalar("loss", complete_loss.item(), step_iter)
            writer.add_scalar("volume_loss", debug_for_volumeloss, step_iter)

            if args['grad_lambda'] > 0:
                writer.add_scalar("grad_loss", grad_loss.item(), step_iter)
            if args['dropout_technique']:
                if args['dropout_technique'] == 'smallify':
                    writer.add_scalar("beta_loss", loss_Betas, step_iter)
                    writer.add_scalar("nw_weight_loss", loss_Weights, step_iter)
                if args['dropout_technique'] == 'variational':
                    writer.add_scalar("Log_Likelyhood", ll, step_iter)
                    writer.add_scalar("DKL", dkl, step_iter)
                    writer.add_scalar("nw_weight_loss", loss_Weights, step_iter)

            # M: Stop training, if we reach max amount of passes over volume
            if (int(volume_passes) + 1) == args['max_pass']:
                break
    return model


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
                           args['pruning_momentum'], use_resnet=args['use_resnet'],
                           pruning_threshold= args['pruning_threshold'],
                           variational_init_droprate=args['variational_init_droprate'])
    model.to(device)
    model.train()

    # M: Setup loss, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    lrStrategy = lrdecay.LearningRateDecayStrategy.create_instance(args, optimizer)
    loss_criterion = torch.nn.MSELoss().to(device)

    #mlflow.start_run(experiment_id=get_Mlfow_Experiment(args['expname']))
    global writer
    if args['Tensorboard_log_dir']:
        writer = SummaryWriter(args['Tensorboard_log_dir'])
    else:
        writer = SummaryWriter('runs/'+args['expname'])

    if args['dropout_technique']:
        args_first = deepcopy(args)
        #args_first['max_pass'] *= (2.0/3.0)

        model = solveModel(model, optimizer, lrStrategy, loss_criterion, volume,
                           dataset, data_loader, args_first, verbose)

        # M: remove dropout layers from model
        if args['dropout_technique'] == 'smallify':
            sign_variance_pruning_strategy_do_prune(model, device, args['pruning_threshold'])  # M: pre-prune
            if args['use_resnet']:
                model = prune_smallify_use_resnet(model, SmallifyDropout)  # M: prune and mult betas
            else:
                model = prune_smallify_no_Resnet(model, SmallifyDropout)  # M: prune and mult betas

        if args['dropout_technique'] == 'variational':
            if args['use_resnet']:
                model = prune_variational_dropout_use_resnet(model)
                #pass
            else:
                model = prune_variational_dropout_no_resnet(model)
                #pass
        model.to(device)

        # M: finetuning after pruning
        #print('---Retraining after pruning---')
        #args_second = deepcopy(args)
        #args_second['max_pass'] *= (1.0/3.0)
        #args_second['dropout_technique'] = ''
        #optimizer = torch.optim.Adam(model.parameters(), lr=args['lr']/100.0)
        #model = solveModel(model, optimizer, lrStrategy, loss_criterion, volume, dataset,
        #                   data_loader, args_second, verbose)
    else:
        model = solveModel(model, optimizer, lrStrategy, loss_criterion, volume, dataset,
                           data_loader, args, verbose)

    info = gather_training_info(model, dataset, volume, args, verbose)
    #mlflow.end_run()
    writer.close()
    return info


