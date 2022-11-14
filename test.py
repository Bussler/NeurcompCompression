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


def emaTest2():

    #momentum = 0.5
    momentum = 2/(100+1)
    print("Momentum: ",momentum)

    threshold = 0.9

    breaks = []

    for j in range(100):

        oldEMA = 1
        oldEMAVar = 0

        numbers = [1]

        for i in range(1000):
            #rand = np.random.randint(2, size=1)[0]
            rand = random.choices([1, -1], cum_weights=(0.7, 1.0), k=1)[0]
            newVal = rand

            numbers.append(newVal)

            phi_i = newVal - oldEMA
            oldEMA = oldEMA + (momentum * phi_i)
            oldEMAVar = (1.0-momentum) * (oldEMAVar + (momentum*(phi_i ** 2)))

            #print("EMA: ", oldEMAVar)
            if oldEMAVar > threshold:
                #print('DAB ', i)
                #print(numbers)
                breaks.append(i)
                break

    b = np.asarray(breaks)
    print("Mean break: ", np.mean(b))


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


def plotTest():
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


if __name__ == '__main__':
    #emaTest2()
    #calculate_variance_of_data()
    #psnr_test()
    plotTest()