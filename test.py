import numpy as np
import torch
import collections
from data.IndexDataset import get_tensor, IndexDataset
from model.model_utils import setup_neurcomp
from visualization.OutputToVTK import tiled_net_out
import visualization.pltUtils as pu
import random
import os
from model.model_utils import compute_num_neurons
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from itertools import product
import tikzplotlib


def emaTest2():
    experimentNames = np.linspace(0, 39, 40, dtype=int)
    experimentNames = np.delete(experimentNames, 8, axis=0)
    experimentNames = np.delete(experimentNames, 8, axis=0)
    pass



def calculate_variance_of_data():
    volume_mhd_p = get_tensor('datasets/mhd1024.h5')
    volume_Ejecta = get_tensor('datasets/Ejecta/snapshot_070_256.cvol')

    variance_mhd_p = torch.var(volume_mhd_p).item()
    variance_Ejecta = torch.var(volume_Ejecta).item()

    print('Variance mhd_p:', variance_mhd_p)  # 0.030741851776838303
    print('Standard Deviation mhd_p:', np.sqrt(variance_mhd_p))  # 0.1753335443571432

    print('Variance Ejecta:', variance_Ejecta)  # 0.02090180665254593
    print('Standard Deviation Ejecta:', np.sqrt(variance_Ejecta))  # 0.14457457125146844


def psnr_test():
    modelPath = '/home/bussler/stash/Masterarbeit/NeurcompCompression/experiments/test_experiment50/model.pth'
    featureList = [68, 68, 68, 68, 68, 68, 68, 68]
    dataPath = 'datasets/test_vol.npy'

    model = setup_neurcomp(0, 0, 0,
                           3, 1, 30, modelPath,
                           featureList=featureList)

    volume = get_tensor(dataPath)
    dataset = IndexDataset(volume, 16)

    psnr, l1_diff, mse, rmse = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True,
                                             write_vols=False)


def Hyperparam_Best_Runs():
    BASENAMEPruned = 'experiments/hyperparam_search/test_experiment_smallify_GridSearch/test_experimentHyperSearch'
    # 'experiments/hyperparam_search/test_experiment_smallify_RandomSearch/test_experimentHyperSearch'
    # 'experiments/hyperparam_search/mhd_p_Random_betas/mhd_p_HyperSearch'
    # experimentNamesPruned = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    experimentNamesPruned = []
    for i in range(0, 54):
        experimentNamesPruned.append(i)

    BASENAMEUnpruned = 'experiments/diff_comp_rates/test_experiment_diff_comp_rates/test_experimentComp'
    # 'experiments/diff_comp_rates/mhd_p_diffCompRates/mhd_p_'
    experimentNamesUnpruned = [20, 50, 100, 150, 200, 300, 400]

    QUANTNAMECONFIG = 'Dequant_Info.txt'

    compressionRatioPruned = []
    PSNRPruned = []
    rmsePruned = []

    # M: generate lists...
    pu.generate_plot_lists(([compressionRatioPruned, PSNRPruned, rmsePruned],),
                        (['Quant_Compression_Ratio', 'psnr', 'rmse'],),
                        BASENAMEPruned, (QUANTNAMECONFIG,), experiment_names=experimentNamesPruned)

    largestGain = 0
    largestExp = 0
    for i in range(0, 54):
        if PSNRPruned[i] > 41.0:
            print(i, ' Ratio: ', compressionRatioPruned[i] / rmsePruned[i])
            if compressionRatioPruned[i] > largestGain:
                largestGain = compressionRatioPruned[i]
                largestExp = i

    print("Most Compression: ",largestGain, " At: ", largestExp)

    pass

def calcParetoStuff():
    BASENAME = 'experiments/diff_comp_rates/mhd_Baselines/OptimizeCompression_WithFeaturesPerLayer/mhd_p_'
    experimentNames = np.linspace(0, 59, 60, dtype=int)
    #experimentNames = np.delete(experimentNames, 5, axis=0)
    #experimentNames = np.delete(experimentNames, 5, axis=0)
    #experimentNames = np.delete(experimentNames, 8, axis=0)

    InfoName = 'info.txt'
    configName = 'config.txt'

    PSNR = []
    CompressionRatio = []

    pu.generate_plot_lists(([PSNR, CompressionRatio],),
                        (['psnr', 'compression_ratio'],),
                        BASENAME, (InfoName,), experiment_names=experimentNames)

    d = pu.findParetoValues(CompressionRatio, PSNR, BASENAME, experimentNames)
    pass

def analyse_NW_Weights():
    from torch.nn import Linear
    from model.VariationalDropoutLayer import VariationalDropout

    Configpath = 'experiments/Test_DiffDropratesPerLayer/Unpruned_Net_TestSet_WithEntropy/config.txt'
    #Configpath = 'experiments/diff_comp_rates/mhd_p_Baselines/100/mhd_p_211/config.txt'
    #Configpath = 'experiments/hyperparam_search/mhd_p_Variational_NAS/100/mhd_p_100_39/config.txt'
    args = pu.dict_from_file(Configpath)
    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])

    model = setup_neurcomp(args['compression_ratio'], dataset.n_voxels, args['n_layers'],
                           args['d_in'], args['d_out'], args['omega_0'], args['checkpoint_path'],
                           dropout_technique=args['dropout_technique'],
                           featureList=args['feature_list'],)

    layers = []
    for layer in model.net_layers.modules():
        #if isinstance(layer, Linear):
        #    layers.append(layer.weight.data)
        if isinstance(layer, VariationalDropout):
            layers.append(layer.dropout_rates.detach().data.reshape(-1).numpy())

    fig, ax = plt.subplots(nrows=len(layers), ncols=1, figsize=(8, 8))
    fig.tight_layout()

    for i in range(len(layers)):
        ax[i].hist(layers[i], bins=120, label=str(i))  # range=(-0.5, 0.5)
        ax[i].title.set_text('Layer '+str(i))

    filepath = 'plots/LatexFigures/Var_Droprate_Analysis/' + 'mhd_p_' + 'Unpruned_Net_TestSet_WithEntropy' + 'Weight_Historgramm'
    plt.savefig(filepath + '.png')
    tikzplotlib.save(filepath + '.pgf')

    pass


def test_different_dropout_rates():
    from model.VariationalDropoutLayer import VariationalDropout
    from model.pruning import prune_variational_dropout_use_resnet
    from model.model_utils import write_dict, write_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ConfigPath = 'experiments/Test_DiffDropratesPerLayer/Unpruned_Net_TestSet_WithEntropy/config.txt'

    args = pu.dict_from_file(ConfigPath)

    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])

    pruning_threshold_list = []
    numruns = 1000
    results = []

    for i in range(numruns):

        n1 = random.uniform(0.5, 0.9)
        n2 = random.uniform(0.4, 0.99)
        n3 = random.uniform(0.2, 0.99)

        pruning_threshold_list = [n1, n2, n3]

        VariationalDropout.i = 0
        VariationalDropout.set_feature_list(pruning_threshold_list)

        model = setup_neurcomp(args['compression_ratio'], dataset.n_voxels, args['n_layers'],
                               args['d_in'], args['d_out'], args['omega_0'], args['checkpoint_path'],
                               dropout_technique=args['dropout_technique'],
                               featureList=args['feature_list'],)
        model.to(device)

        model = prune_variational_dropout_use_resnet(model)
        model.to(device)

        psnr, l1_diff, mse, rmse = tiled_net_out(dataset, model, True, gt_vol=volume.cpu(), evaluate=True,
                                                 write_vols=False)
        num_net_params = 0
        for layer in model.parameters():
            num_net_params += layer.numel()
        compression_ratio = dataset.n_voxels / num_net_params
        compr_rmse = compression_ratio / rmse

        info = {}
        info['pruning_threshold_list'] = pruning_threshold_list
        info['psnr'] = psnr
        info['compression_ratio'] = compression_ratio
        info['compr_rmse'] = compr_rmse

        results.append(info)

    write_list(results, "experiments/Test_DiffDropratesPerLayer/Unpruned_Net_TestSet_WithEntropy/Results.txt")


def create_parallel_coordinates():
    import plotly.graph_objects as go
    import plotly.express as px
    import ast

    filename = 'experiments/Test_DiffDropratesPerLayer/Unpruned_Net_TestSet_WithEntropy/Results.txt'
    file = open(filename, 'r')
    Lines = file.readlines()

    # create data
    data = []
    d1 = []
    d2 = []
    d3 = []
    psnr = []
    compr = []
    x= np.linspace(0, 999, 1000, dtype=int)
    for line in Lines:
        d = ast.literal_eval(line)
        data.append(d)
        d1.append(d['pruning_threshold_list'][0])
        d2.append(d['pruning_threshold_list'][1])
        d3.append(d['pruning_threshold_list'][2])
        psnr.append(d['psnr'])
        compr.append(d['compression_ratio'])

    df = {'id': x,
          'Threshold Layer 1': d1,
          'Threshold Layer 2': d2,
          'Threshold Layer 3': d3,
          'PSNR': psnr,
          'Compression Ratio': compr}

    #fig = px.parallel_coordinates(df, color="id",
    #                              color_continuous_scale=px.colors.diverging.Tealrose,
    #                              color_continuous_midpoint=1)

    fig = go.Figure(data=
    go.Parcoords(
        line=dict(color=df['id'],
                  colorscale=[[0, 'purple'], [0.5, 'lightseagreen'], [1, 'gold']]),
        dimensions=list([
            dict(#range=[0.55, 0.95],
                 label='Threshold Layer 1', values=df['Threshold Layer 1']),
            dict(#range=[0.5, 0.9],
                 label='Threshold Layer 2', values=df['Threshold Layer 2']),
            dict(#range=[0.8, 1.0],
                 label='Threshold Layer 3', values=df['Threshold Layer 3']),
            dict(#range=[23.8, 24.0],
                 #constraintrange=[39.08, 40.0],
                 label='PSNR', values=df['PSNR']),
            dict(#range=[100.0, 162.0],
                 constraintrange=[120.0, 300.0],
                 label='Compression Ratio', values=df['Compression Ratio'])
        ])
    )
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    #filename = 'plots/test'
    filename = 'plots/LatexFigures/Var_Droprate_Analysis/ParallelCoordPlots/testvol_Unpruned_WithEntropy_Parallel_Coordinates_ConstrainCompr'
    fig.write_image(filename+'.png')
    tikzplotlib.save(filename + '.pgf')


if __name__ == '__main__':
    #emaTest2()
    #calculate_variance_of_data()
    #psnr_test()
    #Hyperparam_Best_Runs()

    #calcParetoStuff()
    #analyse_NW_Weights()
    #test_different_dropout_rates()
    create_parallel_coordinates()

    #testingstuff()
