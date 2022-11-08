import numpy as np
import torch
import collections
from data.IndexDataset import get_tensor, IndexDataset
from model.model_utils import setup_neurcomp
from visualization.OutputToVTK import tiled_net_out



def ema_test():
    momentum = 50

    signs = collections.deque(maxlen=momentum)
    for val in np.random.randint(2, size=momentum):
        if val ==1:
            signs.append(1)
        if val ==0:
            signs.append(-1)

    old_variance_ema = 0.0

    #newMean = oldmean - ((testq[0]-oldmean)/3) + ((4-oldmean)/3)

    oldVar = 0
    oldMean = 1

    n = 50

    for i in range(1000):

        rand = np.random.randint(2, size=1)[0]
        newVal = 0.0
        if rand ==1:
            signs.append(1)
            newVal = 1
        if rand ==0:
            signs.append(-1)
            newVal = -1

        oldVar = (n / (n+1)) * (oldVar + (((oldMean-newVal) ** 2) / (n+1)))
        oldMean = oldMean + ((newVal - oldMean) / (n + 1))
        n += 1

        #var = np.var(signs)
        smoother = (0.5)
        old_variance_ema = oldVar * smoother + old_variance_ema * (1-smoother)

        print("EMA: ", old_variance_ema, " variance: ", oldVar)
        if old_variance_ema > 0.5:
            print('DAB ', i)
            break


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

if __name__ == '__main__':
    #ema_test()
    #calculate_variance_of_data()
    psnr_test()