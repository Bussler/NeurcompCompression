from cProfile import label
from mlflow import log_metric, log_param, log_artifacts
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pltUtils import generate_array_MLFlow
import numpy as np


def QuantBitsExperiment():
    
    usedBits20 = [3, 5, 7, 9, 11]
    ComprAfter20 = [189.70799021949918, 118.33178479392738, 83.82906322574236, 60.230213259569915, 38.3810585211636]
    PSNR20 = [25.0941104888916, 35.74490737915039, 44.80397033691406, 50.74647521972656, 46.3750114440918]
    rmse20 = [0.11125625669956207, 0.03264258801937103, 0.011503537185490131, 0.005803719162940979, 0.009600180201232433]

    usedBits50 = [3, 5, 7, 9, 11]
    ComprAfter50 = [ 450.93192598035944, 284.4141069397042, 195.57847767507894, 126.50636280150682, 64.61989143858239]
    PSNR50 = [19.78584861755371, 32.489830017089844, 42.76970291137695, 46.63974380493164, 46.99079513549805]
    rmse50 = [0.20499229431152344, 0.04748303070664406, 0.014539340510964394, 0.009311995469033718, 0.008943142369389534]

    usedBits100 = [3, 5, 7, 9, 11]
    ComprAfter100 = [ 831.3831752678901, 526.3978788115106, 345.3922120452336, 196.63821481632533, 82.95746432829034]
    PSNR100 = [19.28512191772461, 29.507320404052734, 37.62372970581055, 39.94763946533203, 40.010765075683594]
    rmse100 = [0.2171570509672165, 0.06693664938211441, 0.026293201372027397, 0.020120924338698387, 0.01997522823512554]

    usedBits200 = [3, 5, 7, 9, 10]
    ComprAfter200 = [ 1474.7651299978152, 928.2178217821782, 563.2980055078027, 272.6391469424025, 168.32078200588498]
    PSNR200 = [15.7962, 25.1166, 25.9599, 28.3524, 33.1816]
    rmse200 = [0.3245, 0.1110, 0.1007, 0.0765, 0.0438]

    usedBits20Ejecta = [3, 5, 7, 9, 10]
    ComprAfter20Ejecta = [202.377741991, 123.97217183118366, 88.87720143985888, 68.13400828871195, 60.03580559914261]
    PSNR20Ejecta = [36.817108154296875, 45.5382766723, 50.15935897827148, 50.74647521972656, 50.79027557373047]
    rmse20Ejecta = [0.02885191, 0.01057100296020, 0.0062095788307487965, 0.0057954974472522736, 0.005774522665888071]

    usedBits50Ejecta = [3, 5, 7, 9, 10]
    ComprAfter50Ejecta = [ 490.8704594993929, 303.57210581551044, 216.8677701440639, 162.11593502691107, 138.61916368533682]
    PSNR50Ejecta = [34.81428146362305,43.47317123413086, 47.4228363037109, 48.30357360839844, 48.69401931762695]
    rmse50Ejecta = [0.03633422777056694, 0.013408228754997253, 0.00850918609648943, 0.0076886676251888275, 0.007350706495344639]

    usedBits100Ejecta = [3, 5, 7, 9, 10]
    ComprAfter100Ejecta = [ 959.5479424633246, 598.6197348937612, 423.9186385860296, 304.1637462947687, 248.81121764212992]
    PSNR100Ejecta = [33.60960388183594, 41.51560974121094, 45.95808410644531, 46.493072509765625, 46.52091979980469]
    rmse100Ejecta = [0.04173974320292473, 0.016797691583633423, 0.01007223036140203, 0.009470577351748943, 0.00944026280194521]

    usedBits200Ejecta = [3, 5, 7, 9, 10]
    ComprAfter200Ejecta = [ 1833.876154560857, 1153.9853492451077, 801.3381415231772, 534.8768909505364, 408.5875089804318]
    PSNR200Ejecta = [30.531883239746094, 39.06645965576172, 42.76048278808594, 43.11998748779297, 43.150421142578125]
    rmse200Ejecta = [0.05948888510465622, 0.02226932905614376, 0.014554785564541817, 0.013964668847620487, 0.013915829360485077]

    plt.plot(usedBits20Ejecta, np.asanyarray(ComprAfter20Ejecta) / 20, label='Orig Compression 20')
    plt.plot(usedBits50Ejecta, np.asanyarray(ComprAfter50Ejecta) / 50, label='Orig Compression 50')
    plt.plot(usedBits100Ejecta, np.asanyarray(ComprAfter100Ejecta) / 100, label='Orig Compression 100')
    plt.plot(usedBits200Ejecta, np.asanyarray(ComprAfter200Ejecta) / 200, label='Orig Compression 200')

    #plt.plot(usedBits20Ejecta, PSNR20Ejecta, label='Orig Compression 20')
    #plt.plot(usedBits50Ejecta, PSNR50Ejecta, label='Orig Compression 50')
    #plt.plot(usedBits100Ejecta, PSNR100Ejecta, label='Orig Compression 100')
    #plt.plot(usedBits200Ejecta, PSNR200Ejecta, label='Orig Compression 200')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Quant Bits')
    plt.ylabel('Compression_Gain')
    plt.legend()
    
    filepath = 'plots/quantization/'+'Quantization_Ejecta_CompressionGain.png'
    plt.savefig(filepath)


def QuantVsOrigExperiment():
    NeurcompQuantPSNR = [48.90, 46.59, 39.9074, 31.3230, 27.9176]
    NeurcompQuantRMSE = [0.0072, 0.0094, 0.0202, 0.0543, 0.0804]
    NeurcompQuantCompRate = [60.230, 126.506, 196.638, 242.587, 272.6391469424025]

    comprRates = ['20','50','100','150','200']
    comprIdsMSE = ['5','10','11','12','13']

    param = 'rmse'
    mlClient = MlflowClient()

    allRunsDataMSE, realCompRatesMSE = generate_array_MLFlow(mlClient, comprIdsMSE, param)

    plt.plot(realCompRatesMSE, np.asanyarray(NeurcompQuantCompRate) / np.asanyarray(realCompRatesMSE), label='Neurcomp Testscene')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Original Compression Rate')
    plt.ylabel('Quantization Compression Gain')
    plt.legend()
    
    filepath = 'plots/'+'test_experiment_QuantizationGain.png'
    plt.savefig(filepath)



def rmseTTHRESHExperiment():
    baseExpname = 'test_experimentComp'
    comprRates = ['20','50','100','150','200']
    comprIds = ['5','10','11','12','13']
    comprIdsEjecta = ['19','20','21','22','23']
    
    tthreshPSNR = [46.0206, 40.9151, 38.4164, 29.1186, 27.3306]
    tthreshRMSE = [0.00433, 0.00881, 0.0114, 0.0350, 0.0496]
    tthreshCompRate = [90.9495, 150.344, 179.269, 520.632, 689.549]

    NeurcompQuantPSNR = [48.90, 46.59, 39.9074, 31.3230, 27.9176]
    NeurcompQuantRMSE = [0.0072, 0.0094, 0.0202, 0.0543, 0.0804]
    NeurcompQuantCompRate = [60.230, 126.506, 196.638, 242.587, 272.6391469424025]

    # RMSE target
    tthreshPSNREjecta = [44.583, 42.1581, 40.6303, 38.6363, 33.9794]
    tthreshRMSEEjecta = [0.005638692785004683, 0.007525124659809455, 0.00890579221394341, 0.011315378855451252, 0.019798166356808308]
    tthreshCompRateEjeta = [99.914, 182.523, 300.772, 455.698, 1747.54]

    # PSNR target
    #tthreshPSNREjecta = [50.6472, 48.2054, 46.6871,44.629, 38.1792]
    #tthreshRMSEEjecta = [0.00293522, 0.0036583038694397536, 0.00431510500617772, 0.00561060496130343, 0.01184393076485875]
    #tthreshCompRateEjeta = [31.7948, 48.6868, 69.6177, 98.9255, 527.246]

    NeurcompQuantPSNREjecta9 = [51.2516, 48.8107, 47.2383, 45.1975, 38.1867]
    NeurcompQuantRMSEEjecta9 = [0.0055, 0.0073, 0.0087, 0.0110, 0.0246]
    NeurcompQuantCompRateEjecta9 = [68.13400828871195, 162.11593502691107, 304.1637462947687, 534.8768909505364, 852.1111280410381]

    NeurcompQuantPSNREjecta7 = [50.6472, 48.2054, 46.6871, 44.6290, 38.1792]
    NeurcompQuantRMSEEjecta7 = [0.0059, 0.0078, 0.0093, 0.0117, 0.0247]
    NeurcompQuantCompRateEjecta7 = [88.87720143985888, 216.8677701440639, 423.9186385860296, 801.3381415231772, 1426.4520681885813]

    param = 'rmse'#'psnr', 'rmse'

    mlClient = MlflowClient()
    expList = mlClient.search_experiments()

    allRunsData = []
    realCompRates = []
    for id in comprIdsEjecta:
        runs = mlClient.search_runs(experiment_ids=id)
        bestrun = None
        for entry in runs:
            if bestrun is None or entry.data.params[param] < bestrun.data.params[param]:
                bestrun = entry
        allRunsData.append(float(bestrun.data.params[param]))
        realCompRates.append(float(bestrun.data.params['compression_ratio']))

    #plt.plot(realCompRates, allRunsData, label='Neurcomp no Quant')
    plt.plot(tthreshCompRateEjeta, tthreshRMSEEjecta, label='TTHRESH')
    #plt.plot(NeurcompQuantCompRateEjecta9, NeurcompQuantRMSEEjecta9, label='Neurcomp Quant 9 bits')
    plt.plot(NeurcompQuantCompRateEjecta7, NeurcompQuantRMSEEjecta7, label='Neurcomp Quant 7 bits')

    #plt.plot(tthreshCompRateEjeta, tthreshPSNREjecta, label='TTHRESH')
    #plt.plot(NeurcompQuantCompRateEjecta9, NeurcompQuantPSNREjecta9, label='Neurcomp Quant 9 bits')
    #plt.plot(NeurcompQuantCompRateEjecta7, NeurcompQuantPSNREjecta7, label='Neurcomp Quant 7 bits')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Compression Rate')
    plt.ylabel(param)
    plt.legend()
    
    filepath = 'plots/'+'Ejecta_TTHRESH_'+param+'.png'
    plt.savefig(filepath)

def paramExperiment():
    comprRates = ['20','50','100','150','200']
    comprIdsMSE = ['5','10','11','12','13']

    comprIdsL1 = ['14','15','16','17','18']

    comprIdsEjecta = ['19','20','21','22','23']

    param = 'rmse'

    mlClient = MlflowClient()

    allRunsDataMSE, realCompRatesMSE = generate_array_MLFlow(mlClient, comprIdsEjecta, param)

    plt.plot(realCompRatesMSE, allRunsDataMSE)

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Compression Rate')
    plt.ylabel(param)
    #plt.legend()
    
    filepath = 'plots/'+'Ejecta_'+param+'.png'
    plt.savefig(filepath)

if __name__ == '__main__':
    rmseTTHRESHExperiment()
    #paramExperiment()
    #QuantBitsExperiment()
    #QuantVsOrigExperiment()
