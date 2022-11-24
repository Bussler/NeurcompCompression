from cProfile import label
from mlflow import log_metric, log_param, log_artifacts
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pltUtils import generate_array_MLFlow, dict_from_file, append_lists_from_dicts, generate_plot_lists,\
    normalize_array_0_1, normalize_array, generate_orderedValues, generateMeanValues
import numpy as np
from itertools import product


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


def OrigVSSelfImplmentation():
    Test_OrigPSNR = [47.1365, 39.9855, 33.5493, 30.7486, 28.6885]
    Test_OrigRMSE = [0.0088, 0.0200, 0.0420, 0.0580, 0.0736]
    Test_OrigCompRate = [51.11390449650911, 101.7516355633272, 200.52284475075754, 314.8320895522388, 430.2651708312086]

    Test_OrigQuantPSNR = [46.8316, 39.9078, 33.5359, 30.7470, 28.6870]
    Test_OrigQuantRMSE = [0.0091, 0.0202, 0.0421, 0.0580, 0.0736]
    Test_OrigQuantCompRate = [126.4459326558329, 196.49224947238193, 272.3586256985494, 319.03580290677064, 347.39198682483726]

    mhd_p_OrigPSNR = [50.3264, 45.6230, 40.9, 37.0811]
    mhd_p_OrigRMSE = [0.0061, 0.0105, 0.018, 0.0280]
    mhd_p_OrigCompRate = [50.15024165693788, 100.28107215646905, 201.45275729264114, 411.1016760053553]

    mhd_p_OrigQuantPSNR = [50.1650, 45.5244, 40.8, 37.0627]
    mhd_p_OrigQuantRMSE = [0.0062, 0.0106, 0.019, 0.0280]
    mhd_p_OrigQuantCompRate = [162.12500091664407, 300.5437614699685, 528.4184612442936, 860.7999896172665]

    Test_MyPSNR = [46.993186950683594, 40.01081466674805, 33.18212890625, 30.62228775024414, 28.301376342773438]
    Test_MyRMSE = [0.008940679021179676, 0.01997510902583599, 0.04384535178542137, 0.058872923254966736, 0.07690615952014923]
    Test_MyCompRate = [51.11390449650911, 101.7516355633272, 200.52284475075754, 314.8320895522388, 430.2651708312086]

    Test_MyQuantPSNR = [46.6522, 39.9328, 33.1472, 29.6226, 28.3016]
    Test_MyQuantRMSE = [0.0093, 0.0202, 0.0440, 0.0661, 0.0769]
    Test_MyQuantCompRate = [126.4459326558329, 196.49224947238193, 272.3586256985494, 319.03580290677064, 347.39198682483726]

    mhd_p_MyPSNR = [50.343326568603516, 45.57869338989258, 40.84169387817383, 37.342159271240234]
    mhd_p_MyRMSE = [0.0060794418677687645, 0.010521928779780865, 0.018152866512537003, 0.027159517630934715]
    mhd_p_MyCompRate = [50.15024165693788, 100.28107215646905, 201.45275729264114, 411.1016760053553]

    mhd_p_MyQuantPSNR = [50.1753, 45.47528076171875, 40.79093551635742, 37.3199]
    mhd_p_MyQuantRMSE = [0.0062, 0.010647944174706936, 0.018259266391396523, 0.0272]
    mhd_p_MyQuantCompRate = [162.14521452145215, 300.6132327746404, 528.6332552245229, 861.3701298701299]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

    ax1.plot(Test_OrigCompRate, Test_OrigPSNR, label='Original TestVolume')
    ax1.plot(Test_MyCompRate, Test_MyPSNR, label='My Implementation TestVolume')
    ax1.plot(mhd_p_OrigCompRate, mhd_p_OrigPSNR, label='Original MHD_P')
    ax1.plot(mhd_p_MyCompRate, mhd_p_MyPSNR, label='My Implementation MHD_P')
    ax1.set_ylabel('psnr')
    ax1.set_xlabel('Compression Rate')
    ax1.title.set_text('Without Quantization')

    ax2.plot(Test_OrigCompRate, Test_OrigRMSE, label='Original TestVolume')
    ax2.plot(Test_MyCompRate, Test_MyRMSE, label='My Implementation TestVolume')
    ax2.plot(mhd_p_OrigCompRate, mhd_p_OrigRMSE, label='Original MHD_P')
    ax2.plot(mhd_p_MyCompRate, mhd_p_MyRMSE, label='My Implementation MHD_P')
    ax2.set_ylabel('rmse')
    ax2.set_xlabel('Compression Rate')

    ax3.plot(Test_OrigQuantCompRate, Test_OrigQuantPSNR, label='Original TestVolume')
    ax3.plot(Test_MyQuantCompRate, Test_MyQuantPSNR, label='My Implementation TestVolume')
    ax3.plot(mhd_p_OrigQuantCompRate, mhd_p_OrigQuantPSNR, label='Original MHD_P')
    ax3.plot(mhd_p_MyQuantCompRate, mhd_p_MyQuantPSNR, label='My Implementation MHD_P')
    ax3.set_ylabel('psnr')
    ax3.set_xlabel('Compression Rate')
    ax3.title.set_text('With Quantization')

    ax4.plot(Test_OrigQuantCompRate, Test_OrigQuantRMSE, label='Original TestVolume')
    ax4.plot(Test_MyQuantCompRate, Test_MyQuantRMSE, label='My Implementation TestVolume')
    ax4.plot(mhd_p_OrigQuantCompRate, mhd_p_OrigQuantRMSE, label='Original MHD_P')
    ax4.plot(mhd_p_MyQuantCompRate, mhd_p_MyQuantRMSE, label='My Implementation MHD_P')
    ax4.set_ylabel('rmse')
    ax4.set_xlabel('Compression Rate')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    
    filepath = 'plots/OrigVSMyImplementation.png'
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
    tthreshPSNREjecta = [51.038630986361255, 48.52475445972449, 47.06839826504262, 44.92355607499145, 43.978323450645114, 42.16238575491416, 40.99271445293446]
    tthreshRMSEEjecta = [0.0056117516941404065, 0.007495356133233042, 0.008863598644750986, 0.011346245910307466, 0.012650678624318219, 0.015592318877128872, 0.01783997629700317]
    tthreshCompRateEjeta = [98.9605, 181.276, 297.497, 459.492, 652.493, 961.858, 1442.02]

    # PSNR target
    #tthreshPSNREjecta = [51.3853, 48.7344, 47.3002, 45.0356, 39.5897, 37.1734, 35.0835]
    #tthreshRMSEEjecta = [0.00269608, 0.0036583, 0.00431511, 0.00560041, 0.0104838, 0.0138462, 0.0176128]
    #tthreshCompRateEjeta = [31.7948, 48.6866, 69.6177, 98.5605, 385.059, 744.975, 1415.74]

    NeurcompQuantPSNREjecta9 = [51.2516, 48.8107, 47.2383, 45.1982, 43.9587, 42.9999]
    NeurcompQuantRMSEEjecta9 = [0.0055, 0.0073, 0.0087, 0.0110, 0.0127, 0.0142]
    NeurcompQuantCompRateEjecta9 = [68.13400828871195, 162.11593502691107, 304.1637462947687, 534.8768909505364, 714.076016173654, 852.1111280410381]

    NeurcompQuantPSNREjecta7 = [50.6472, 48.20541000366211, 46.6871, 44.7053, 43.4916, 42.4256]
    NeurcompQuantRMSEEjecta7 = [0.00587, 0.00777, 0.00926, 0.0116, 0.0134, 0.0151] 
    NeurcompQuantCompRateEjecta7 = [88.87720143985888, 216.8677701440639, 423.9186385860296, 801.3381415231772, 1137.0144014096438, 1426.4520681885813]

    NeurcompQuantPSNRmhd_p9 = [51.56679153442383, 50.18495559692383, 45.4834, 40.7906, 37.31993103027344]
    NeurcompQuantRMSEmhd_9 = [0.005280687473714352, 0.0061913058161735535, 0.0106, 0.0183, 0.02722911536693573]
    NeurcompQuantCompRatemhd_9 = [ 68.412068076328, 162.14521452145215, 300.6132327746404, 528.6332552245229, 861.3701298701299]

    NeurcompQuantPSNRmhd_p7 = [49.3000, 48.0839, 44.0162, 39.9114, 36.7788]
    NeurcompQuantRMSEmhd_7 = [0.0069, 0.0079, 0.0126, 0.0202, 0.0290]
    NeurcompQuantCompRatemhd_7 = [89.26451706678941, 217.03086347037342, 418.9702222278372, 791.9840947627349, 1452.7861742673151]

    NeurcompQuantPSNRmhd_p_noGrad9 = [57.679969787597656, 50.457481384277344, 45.28264617919922, 41.036048889160156, 37.52974319458008]
    NeurcompQuantRMSEmhd_noGrad9 = [0.002612351207062602,0.006000065244734287, 0.01088673621416092, 0.017751192674040794, 0.02657926082611084]
    NeurcompQuantCompRatemhd_noGrad9 = [68.412068076328, 162.14521452145215, 300.6132327746404, 528.6332552245229, 861.3701298701299]

    # PSNR no Grad target
    #tthreshPSNRmhd_p = [55.1833, 49.1794, 45.2835, 41.0376, 38.5452, 34.7223, 30.342, 28.1689]
    #tthreshRMSEmhd_p = [0.0014869144161891758, 0.003114780095102444, 0.004981052701820832, 0.008235872355390245,  0.0118234, 0.017433496413122113, 0.0304019, 0.0390439]
    #tthreshCompRatemhd_p = [20.7634, 35.9296, 58.0608, 104.395, 172.495, 309.201, 741.548, 1218.99]

    # RMSE
    tthreshPSNRmhd_p = [56.09312771034319, 50.073628444238494, 46.08512253045333, 41.55766999675827, 40.41322974713026, 36.78848713255406, 34.925096746538294, 34.597142893155734]
    tthreshRMSEmhd_p = [0.003135982353123946, 0.006271169994000251, 0.009925990818509228, 0.016716544001841114, 0.019070782289479934, 0.028947138499412675, 0.035873616272853986, 0.03725399491978439]
    tthreshCompRatemhd_p = [36.145, 79.6552, 141.825, 297.357, 341.542, 681.394, 1096.76, 1143.66]

    param = 'rmse'#'psnr', 'rmse'
    datset='mhd_p'

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
    #plt.plot(tthreshCompRateEjeta, tthreshRMSEEjecta, label='TTHRESH')
    #plt.plot(NeurcompQuantCompRateEjecta9, NeurcompQuantRMSEEjecta9, label='Neurcomp Quant 9 bits')
    #plt.plot(NeurcompQuantCompRateEjecta7, NeurcompQuantRMSEEjecta7, label='Neurcomp Quant 7 bits')

    #plt.plot(tthreshCompRateEjeta, tthreshPSNREjecta, label='TTHRESH')
    #plt.plot(NeurcompQuantCompRateEjecta9, NeurcompQuantPSNREjecta9, label='Neurcomp Quant 9 bits')
    #plt.plot(NeurcompQuantCompRateEjecta7, NeurcompQuantPSNREjecta7, label='Neurcomp Quant 7 bits')

    #plt.plot(tthreshCompRatemhd_p, tthreshPSNRmhd_p, label='TTHRESH')
    #plt.plot(NeurcompQuantCompRatemhd_9, NeurcompQuantPSNRmhd_p9, label='Neurcomp Quant 9 bits')
    #plt.plot(NeurcompQuantCompRatemhd_7, NeurcompQuantPSNRmhd_p7, label='Neurcomp Quant 7 bits')
    #plt.plot(NeurcompQuantCompRatemhd_noGrad9, NeurcompQuantPSNRmhd_p_noGrad9, label='Neurcomp Quant 9 bits, no Grad')

    plt.plot(tthreshCompRatemhd_p, tthreshRMSEmhd_p, label='TTHRESH')
    plt.plot(NeurcompQuantCompRatemhd_9, NeurcompQuantRMSEmhd_9, label='Neurcomp Quant 9 bits')
    plt.plot(NeurcompQuantCompRatemhd_7, NeurcompQuantRMSEmhd_7, label='Neurcomp Quant 7 bits')
    plt.plot(NeurcompQuantCompRatemhd_noGrad9, NeurcompQuantRMSEmhd_noGrad9, label='Neurcomp Quant 9 bits, no Grad')

    plt.axhline(y=0.1753335443571432, color='y', linestyle='--')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Compression Rate')
    plt.ylabel(param)
    plt.legend()
    
    filepath = 'plots/'+datset+'_TTHRESH_'+param+'.png'
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


def NumberOfChannelsVSCompression():
    BASENAME = 'experiments/mhd_p_diffCompRates/mhd_p_'
    CONFIGNAME = 'info.txt'
    QUANTNAMECONFIG = 'Dequant_Info.txt'

    CompressionRate = []
    PSNR = []
    RMSE = []
    AmountFeatures = []

    CompressionRateQuant = []
    PSNRQuant = []
    RMSEQuant = []

    # M: generate lists for psnr, rmse...
    generate_plot_lists(([CompressionRate, PSNR, RMSE], [CompressionRateQuant, PSNRQuant, RMSEQuant]),
                        (['compression_ratio', 'psnr', 'rmse'], ['Quant_Compression_Ratio', 'psnr', 'rmse']),
                        BASENAME, (CONFIGNAME, QUANTNAMECONFIG), experiment_names=[20, 50, 100, 200, 400])

    # M generate lists for middle layers
    for compr in [20, 50, 100, 200, 400]:
        config_name = BASENAME + str(compr) + '/' + CONFIGNAME
        info = dict_from_file(config_name)
        AmountFeatures.append(info['network_layer_sizes'][1])


    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 15), dpi= 100)

    ax1.plot(CompressionRate, PSNR, label='No Quantization')
    ax1.plot(CompressionRateQuant, PSNRQuant, label='Quantization')
    ax1.title.set_text('PSNR MHD_P')
    ax1.set_ylabel('psnr')
    ax1.set_xlabel('Compression Rate')

    ax2.plot(CompressionRate, RMSEQuant, label='No Quantization')
    ax2.plot(CompressionRateQuant, RMSEQuant, label='Quantization')
    ax2.title.set_text('RMSE MHD_P')
    ax2.set_ylabel('rmse')
    ax2.set_xlabel('Compression Rate')

    ax3.plot(CompressionRate, AmountFeatures, label='No Quantization')
    ax3.plot(CompressionRateQuant, AmountFeatures, label='Quantization')
    ax3.title.set_text('Amount Middle Features MHD_P')
    ax3.set_ylabel('features')
    ax3.set_xlabel('Compression Rate')

    #plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()

    filepath = 'plots/' + 'CompressionratioVSQuality.png'
    plt.savefig(filepath)


def QuantbitsVSCompressionratio():
    BASENAME50 =  'experiments/diff_comp_rates/test_experiment_QuantbitsVSCompressionratio/Ratio50/test_experiment'#'experiments/diff_comp_rates/mhd_p_QuantbitsVSCompressionratio/Ratio50/mhd_p_50'
    BASENAME100 =  'experiments/diff_comp_rates/test_experiment_QuantbitsVSCompressionratio/Ratio100/test_experiment'#'experiments/diff_comp_rates/mhd_p_QuantbitsVSCompressionratio/Ratio100/test_experiment'
    BASENAME200 = 'experiments/diff_comp_rates/test_experiment_QuantbitsVSCompressionratio/Ratio200/test_experiment'
    CONFIGNAME = 'info.txt'
    QUANTNAMECONFIG = 'Dequant_Info.txt'

    experimentNames50= [50,38,31,26,21,17,12,8,4]#[50,43,36,31,26]
    experimentNames100 = [100,65,51,42,34,27,20,14,7]#[100, 82,70,59,49]
    experimentNames200 = [200,106,75,59,48,38,29,20,11]

    used_bits50 = []
    PSNR50 = []
    size50 = []

    used_bits100 = []
    PSNR100 = []
    size100 = []

    used_bits200 = []
    PSNR200 = []
    size200 = []

    # M: generate lists...
    generate_plot_lists(([used_bits50, PSNR50], ),
                        (['used_bits', 'psnr'], ),
                        BASENAME50, (QUANTNAMECONFIG, ), experiment_names=experimentNames50)

    # M generate lists for middle layers
    for compr in experimentNames50:
        config_name = BASENAME50 + str(compr) + '/' + CONFIGNAME
        info = dict_from_file(config_name)
        size50.append(info['network_layer_sizes'][1])

    generate_plot_lists(([used_bits100, PSNR100], ),
                        (['used_bits', 'psnr'], ),
                        BASENAME100, (QUANTNAMECONFIG, ), experiment_names=experimentNames100)

    # M generate lists for middle layers
    for compr in experimentNames100:
        config_name = BASENAME100 + str(compr) + '/' + CONFIGNAME
        info = dict_from_file(config_name)
        size100.append(info['network_layer_sizes'][1])

    if BASENAME200:
        generate_plot_lists(([used_bits200, PSNR200], ),
                            (['used_bits', 'psnr'], ),
                            BASENAME200, (QUANTNAMECONFIG, ), experiment_names=experimentNames200)

        # M generate lists for middle layers
        for compr in experimentNames200:
            config_name = BASENAME200 + str(compr) + '/' + CONFIGNAME
            info = dict_from_file(config_name)
            size200.append(info['network_layer_sizes'][1])

    c_map1 = plt.get_cmap('Blues') # c=[c_map1(val) for val in normalize_array_0_1(PSNR50)]
    c_map2 = plt.get_cmap('Greens')
    c_map3 = plt.get_cmap('Reds')



    plt.scatter(used_bits50, size50, s = [x for x in generate_orderedValues(PSNR50, 200)],
                c=[c_map1(val) for val in normalize_array(PSNR50, np.min(PSNR50), np.max(PSNR50), 0.3, 1)], label='Compression 126')
    plt.scatter(used_bits100, size100,s = [x for x in generate_orderedValues(PSNR100, 200)],
                c=[c_map2(val) for val in normalize_array(PSNR100, np.min(PSNR100), np.max(PSNR100), 0.3, 1)], label='Compression 196')
    plt.scatter(used_bits200, size200, s = [x for x in generate_orderedValues(PSNR200, 200)],
                c=[c_map3(val) for val in normalize_array(PSNR200, np.min(PSNR200), np.max(PSNR200), 0.3, 1)], label='Compression 272')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('#quant_bits')
    plt.ylabel('#channels')
    plt.legend()

    filepath = 'plots/' + 'test_volume' + '_QuantbitsVSCompressionratio' + '.png'
    plt.savefig(filepath)


def CompressionGainVSPSNR():
    BASENAME = 'experiments/hyperparam_search/mhd_p_Random_betas/mhd_p_HyperSearch'
    experimentNames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    CONFIGNAME = 'info.txt'
    QUANTNAMECONFIG = 'Dequant_Info.txt'

    compressionGain = []
    PSNR = []

    # M: generate lists...
    generate_plot_lists(([compressionGain, PSNR],),
                        (['compression_ratio', 'psnr'],),
                        BASENAME, (CONFIGNAME,), experiment_names=experimentNames)

    NoPruningBasename = 'experiments/diff_comp_rates/mhd_p_diffCompRates/mhd_p_100/'
    noPruningInfo = dict_from_file(NoPruningBasename+'info.txt')


    # M: decrease by base-compression
    for elem in range(len(compressionGain)):
        compressionGain[elem] = compressionGain[elem] - noPruningInfo['compression_ratio']

    plt.scatter(compressionGain, PSNR, label = 'Pruned')

    # M: Baseline
    plt.axhline(y=noPruningInfo['psnr'], color='y', linestyle='--', label='No Prune PSNR')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Compression Gain')
    plt.ylabel('psnr')
    plt.legend()

    filepath = 'plots/' + 'mhd_p_' + 'CompressionGain_VS_PSNR' + '.png'
    plt.savefig(filepath)


def PrunedVSUnpruned():
    BASENAMEPruned = 'experiments/hyperparam_search/test_experiment_smallify_GridSearch/test_experimentHyperSearch'
    #'experiments/hyperparam_search/test_experiment_smallify_RandomSearch/test_experimentHyperSearch'
    #'experiments/hyperparam_search/mhd_p_Random_betas/mhd_p_HyperSearch'
    #experimentNamesPruned = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    experimentNamesPruned = []
    for i in range(0, 54):
        experimentNamesPruned.append(i)

    BASENAMEUnpruned = 'experiments/diff_comp_rates/test_experiment_diff_comp_rates_otherHyper/test_experiment'
    #'experiments/diff_comp_rates/mhd_p_diffCompRates/mhd_p_'
    experimentNamesUnpruned = [50,100,150,200,300]

    QUANTNAMECONFIG = 'Dequant_Info.txt'

    compressionRatioPruned = []
    PSNRPruned = []

    compressionRatioUnpruned = []
    PSNRUnpruned = []

    # M: generate lists...
    generate_plot_lists(([compressionRatioPruned, PSNRPruned],),
                        (['Quant_Compression_Ratio', 'psnr'],),
                        BASENAMEPruned, (QUANTNAMECONFIG,), experiment_names=experimentNamesPruned)

    generate_plot_lists(([compressionRatioUnpruned, PSNRUnpruned],),
                        (['Quant_Compression_Ratio', 'psnr'],),
                        BASENAMEUnpruned, (QUANTNAMECONFIG,), experiment_names=experimentNamesUnpruned)


    plt.scatter(compressionRatioPruned, PSNRPruned, label = 'Pruned')
    plt.plot(compressionRatioUnpruned, PSNRUnpruned, label='No Pruning')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Compressionratio after Quantization 8 bits')
    plt.ylabel('psnr')
    plt.legend()

    filepath = 'plots/' + 'test_volume_GridSearch_' + 'PrunedVSUnpruned' + '.png'
    plt.savefig(filepath)


def influenceSmallifyParameter():
    BASENAMEPruned = 'experiments/hyperparam_search/test_experiment_smallify_GridSearch/test_experimentHyperSearch'
    experimentNamesPruned = []
    for i in range(0, 54):
        experimentNamesPruned.append(i)

    CONFIGNAME = 'config.txt'
    QUANTNAMECONFIG = 'Dequant_Info.txt'

    lrPruned = []
    grad_lambdaPruned = []
    lambda_betasPruned = []
    lambda_weightsPruned = []

    compressionRatioPruned = []
    rmsePruned = []

    # M: generate lists...
    generate_plot_lists(([lrPruned, grad_lambdaPruned, lambda_betasPruned, lambda_weightsPruned],
                         [compressionRatioPruned, rmsePruned],),
                        (['lr', 'grad_lambda', 'lambda_betas', 'lambda_weights'],
                         ['Quant_Compression_Ratio', 'rmse'],),
                        BASENAMEPruned, (CONFIGNAME, QUANTNAMECONFIG,), experiment_names=experimentNamesPruned)

    random_search_spaces = {
        "lr": ([0.0001, 0.0007]),
        "grad_lambda": ([1e-04, 1e-05, 1e-06]),
        "lambda_betas": ([3e-04, 3e-05, 3e-06]),
        "lambda_weights": ([1e-06, 1e-05]),
    }
    for instance in product(*random_search_spaces.values()):

        xAxis = []
        compr_rmse = []

        for i in range(0, 54):
            if lrPruned[i] == instance[0] \
                    and lambda_betasPruned[i] == instance[2]\
                    and grad_lambdaPruned[i] == instance[1]\
                    :
                compr_rmse.append(compressionRatioPruned[i] / rmsePruned[i])
                #xAxis.append(lambda_betasPruned[i])
                xAxis.append(lambda_weightsPruned[i])

        plt.plot(xAxis, compr_rmse, label='')

    plt.xlabel('lambda_weight')
    plt.ylabel('compr/rmse')
    filepath = 'plots/' + 'test_volume_GridSearch_' + 'InfluenceParameter_weight' + '.png'
    plt.savefig(filepath)


def influenceHyperSearchParams():
    BASENAMEPruned = 'experiments/hyperparam_search/test_experiment_RandomSearch_Momentum_Threshold/test_experiment_'
    experimentNamesPruned = []
    for i in range(0, 20):
        experimentNamesPruned.append(i)

    CONFIGNAME = 'config.txt'
    infoname = 'info.txt'

    pruning_momentum = []
    pruning_threshold = []

    psnr = []
    rmsePruned = []
    compressionRatioPruned = []

    # M: generate lists...
    generate_plot_lists(([pruning_momentum, pruning_threshold],
                         [psnr, rmsePruned, compressionRatioPruned],),
                        (['pruning_momentum', 'pruning_threshold'],
                         ['psnr', 'rmse', 'compression_ratio'],),
                        BASENAMEPruned, (CONFIGNAME, infoname,), experiment_names=experimentNamesPruned)

    comprRMSE = []
    for i in range(len(compressionRatioPruned)):
        comprRMSE.append(compressionRatioPruned[i] / rmsePruned[i])

    #plt.scatter(pruning_momentum, comprRMSE, label='pruning_momentum')
    plt.scatter(pruning_momentum, comprRMSE, label='pruning_momentum')

    plt.xlabel('pruning_Momentum')
    plt.ylabel('compr/rmse')
    filepath = 'plots/' + 'test_volume_HyperSearch_' + 'Influence_PruningMomentum' + '.png'
    plt.savefig(filepath)


def SmallifyVSNeurcompLR():
    BASENAMESmallify = 'experiments/diff_comp_rates/test_experiment_lrcomparison/Smallify/test_experiment_'
    experimentNamesSmallify = ['50_0', '50_1', '50_2', '100_0', '100_1', '100_2', '200_0', '200_1', '200_2']

    BASENAMENeurcomp = 'experiments/diff_comp_rates/test_experiment_lrcomparison/Neurcomp/test_experiment_'
    experimentNamesNeurcomp = ['50_0', '50_1', '50_2', '100_0', '100_1', '100_2', '200_0', '200_1', '200_2']

    QUANTNAMECONFIG = 'info.txt'

    compressionRatioSmallify = []
    PSNRPrunedSmallify = []

    compressionRatioNeurcomp = []
    PSNRNeurcomp = []

    # M: generate lists...
    generate_plot_lists(([compressionRatioSmallify, PSNRPrunedSmallify],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMESmallify, (QUANTNAMECONFIG,), experiment_names=experimentNamesSmallify)

    generate_plot_lists(([compressionRatioNeurcomp, PSNRNeurcomp],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMENeurcomp, (QUANTNAMECONFIG,), experiment_names=experimentNamesNeurcomp)

    plt.scatter(compressionRatioSmallify, PSNRPrunedSmallify, label='Smallify')
    plt.scatter(compressionRatioNeurcomp, PSNRNeurcomp, label='Neurcomp')

    compressionRatioSmallify_sum = generateMeanValues(compressionRatioSmallify, 3)
    PSNRPrunedSmallify_sum = generateMeanValues(PSNRPrunedSmallify, 3)

    plt.plot(compressionRatioSmallify_sum, PSNRPrunedSmallify_sum, label='Smallify')

    compressionRatioNeurcomp_sum = generateMeanValues(compressionRatioNeurcomp, 3)
    PSNRPrunedneurcomp_sum = generateMeanValues(PSNRNeurcomp, 3)

    plt.plot(compressionRatioNeurcomp_sum, PSNRPrunedneurcomp_sum, label='Neurcomp')

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Compressionratio')
    plt.ylabel('psnr')
    plt.legend()

    filepath = 'plots/' + 'test_volume_' + 'SmallifyVSNeurcompLR' + '.png'
    plt.savefig(filepath)


def SmallifyAftertrainingVSNoAftertrainig():
    #BASENAMERetrain = 'experiments/diff_comp_rates/test_experiment_Retraining/Retrain/test_experiment_'
    BASENAMERetrain = 'experiments/diff_comp_rates/test_experiment_Retraining/4C_Retrain/test_experiment_'
    experimentNamesRetrain = ['50_0', '50_1', '50_2', '100_0', '100_1', '100_2', '200_0', '200_1', '200_2', '300_0', '300_1', '300_2', '400_0', '400_1', '400_2']

    #BASENAMENoRetrain = 'experiments/diff_comp_rates/test_experiment_Retraining/NoRetrain/test_experiment_'
    BASENAMENoRetrain = 'experiments/diff_comp_rates/test_experiment_BroaderNW/4C/test_experiment_'
    experimentNamesNoRetrain = ['50_0', '50_1', '50_2', '100_0', '100_1', '100_2', '200_0', '200_1', '200_2', '300_0', '300_1', '300_2', '400_0', '400_1', '400_2']

    QUANTNAMECONFIG = 'info.txt'

    compressionRatioRetrain = []
    PSNRRetrain = []

    compressionRatioNoRetrain = []
    PSNRNoRetrain = []

    # M: generate lists...
    generate_plot_lists(([compressionRatioRetrain, PSNRRetrain],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMERetrain, (QUANTNAMECONFIG,), experiment_names=experimentNamesRetrain)

    generate_plot_lists(([compressionRatioNoRetrain, PSNRNoRetrain],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMENoRetrain, (QUANTNAMECONFIG,), experiment_names=experimentNamesNoRetrain)

    plt.scatter(compressionRatioRetrain, PSNRRetrain, label='With Retraining')
    plt.scatter(compressionRatioNoRetrain, PSNRNoRetrain, label='No Retraining')

    compressionRatioSmallify_sum = generateMeanValues(compressionRatioRetrain, 3)
    PSNRPrunedSmallify_sum = generateMeanValues(PSNRRetrain, 3)

    plt.plot(compressionRatioSmallify_sum, PSNRPrunedSmallify_sum, label='With Retraining 2 Layers')

    compressionRatioNeurcomp_sum = generateMeanValues(compressionRatioNoRetrain, 3)
    PSNRPrunedneurcomp_sum = generateMeanValues(PSNRNoRetrain, 3)

    plt.plot(compressionRatioNeurcomp_sum, PSNRPrunedneurcomp_sum, label='No Retraining 2 Layers')

    #plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Compressionratio')
    plt.ylabel('psnr')
    plt.legend()

    filepath = 'plots/' + 'test_volume_' + 'SmallifyAftertrainingVSNoAftertrainig' + '.png'
    plt.savefig(filepath)


def SmallifyDifferentNWSizes():

    BASENAMEUnpruned8 = 'experiments/diff_comp_rates/test_experiment_diff_comp_rates_otherHyper/test_experiment'
    experimentNamesUnpruned8 = [50, 100, 200]

    experimentNames = ['50_0', '50_1', '50_2', '100_0', '100_1', '100_2', '200_0', '200_1', '200_2']
    experimentNames2 = ['50_0', '50_1', '50_2', '100_0', '100_1', '100_2', '200_0', '200_1', '200_2', '300_0', '300_1', '300_2', '400_0', '400_1', '400_2']

    BASENAMEUnpruned6 = 'experiments/diff_comp_rates/test_experiment_BroaderNW/6C_NoPrune/test_experiment'
    BASENAMEUnpruned4 = 'experiments/diff_comp_rates/test_experiment_BroaderNW/4C_NoPrune/test_experiment'
    BASENAMEUnpruned2 = 'experiments/diff_comp_rates/test_experiment_BroaderNW/2C_NoPrune/test_experiment'

    BASENAMEPruned8 = 'experiments/diff_comp_rates/test_experiment_BroaderNW/8C/test_experiment_'
    BASENAMEPruned6 = 'experiments/diff_comp_rates/test_experiment_BroaderNW/6C/test_experiment_'
    #BASENAMEPruned4 = 'experiments/diff_comp_rates/test_experiment_BroaderNW/4C/test_experiment_'
    BASENAMEPruned4 = 'experiments/diff_comp_rates/test_experiment_BroaderNW/4C_BiggerWeightLambda/test_experiment_'
    BASENAMEPruned2 = 'experiments/diff_comp_rates/test_experiment_BroaderNW/2C/test_experiment_'

    InfoName = 'info.txt'
    InfoNameDequant = 'Dequant_Info.txt'

    compressionRatioUnpruned8 = []
    PSNRUnpruned8 = []

    compressionRatioUnpruned6 = []
    PSNRUnpruned6 = []

    compressionRatioUnpruned4 = []
    PSNRUnpruned4 = []

    compressionRatioUnpruned2 = []
    PSNRUnpruned2 = []

    compressionRatioPruned8 = []
    PSNRPruned8 = []

    compressionRatioPruned6 = []
    PSNRPruned6 = []

    compressionRatioPruned4_Other = []
    PSNRPruned4_Other = []

    compressionRatioPruned4 = []
    PSNRPruned4 = []

    compressionRatioPruned2 = []
    PSNRPruned2 = []

    # M: generate lists...
    generate_plot_lists(([compressionRatioUnpruned8, PSNRUnpruned8],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEUnpruned8, (InfoName,), experiment_names=experimentNamesUnpruned8)

    generate_plot_lists(([compressionRatioUnpruned6, PSNRUnpruned6],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEUnpruned6, (InfoName,), experiment_names=experimentNames)

    generate_plot_lists(([compressionRatioUnpruned4, PSNRUnpruned4],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEUnpruned4, (InfoName,), experiment_names=experimentNames2)

    generate_plot_lists(([compressionRatioUnpruned2, PSNRUnpruned2],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEUnpruned2, (InfoName,), experiment_names=experimentNames2)

    generate_plot_lists(([compressionRatioPruned8, PSNRPruned8],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEPruned8, (InfoName,), experiment_names=experimentNames)

    generate_plot_lists(([compressionRatioPruned6, PSNRPruned6],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEPruned6, (InfoName,), experiment_names=experimentNames)

    generate_plot_lists(([compressionRatioPruned4, PSNRPruned4],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEPruned4, (InfoName,), experiment_names=experimentNames2)

    generate_plot_lists(([compressionRatioPruned2, PSNRPruned2],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEPruned2, (InfoName,), experiment_names=experimentNames2)

    # M: generate avg values
    #compressionRatioUnpruned8 = generateMeanValues(compressionRatioUnpruned8, 3)
    #PSNRUnpruned8 = generateMeanValues(PSNRUnpruned8, 3)

    compressionRatioUnpruned6 = generateMeanValues(compressionRatioUnpruned6, 3)
    PSNRUnpruned6 = generateMeanValues(PSNRUnpruned6, 3)

    compressionRatioUnpruned4 = generateMeanValues(compressionRatioUnpruned4, 3)
    PSNRUnpruned4 = generateMeanValues(PSNRUnpruned4, 3)

    compressionRatioUnpruned2 = generateMeanValues(compressionRatioUnpruned2, 3)
    PSNRUnpruned2 = generateMeanValues(PSNRUnpruned2, 3)

    compressionRatioPruned8 = generateMeanValues(compressionRatioPruned8, 3)
    PSNRPruned8 = generateMeanValues(PSNRPruned8, 3)

    compressionRatioPruned6 = generateMeanValues(compressionRatioPruned6, 3)
    PSNRPruned6 = generateMeanValues(PSNRPruned6, 3)

    compressionRatioPruned4 = generateMeanValues(compressionRatioPruned4, 3)
    PSNRPruned4 = generateMeanValues(PSNRPruned4, 3)

    compressionRatioPruned2 = generateMeanValues(compressionRatioPruned2, 3)
    PSNRPruned2 = generateMeanValues(PSNRPruned2, 3)

    # M: plot
    #plt.plot(compressionRatioUnpruned8, PSNRUnpruned8, label='Unpruned 8', alpha=0.6)
    #plt.plot(compressionRatioUnpruned6, PSNRUnpruned6, label='Unpruned 6', alpha=0.6)
    plt.plot(compressionRatioUnpruned4, PSNRUnpruned4, label='Unpruned 4', alpha=0.6)
    #plt.plot(compressionRatioUnpruned2, PSNRUnpruned2, label='Unpruned 2', alpha=0.6)

    #plt.plot(compressionRatioPruned8, PSNRPruned8, label='Pruned 8')
    #plt.plot(compressionRatioPruned6, PSNRPruned6, label='Pruned 6')
    plt.plot(compressionRatioPruned4, PSNRPruned4, label='Pruned 4')
    #plt.plot(compressionRatioPruned2, PSNRPruned2, label='Pruned 2')

    #plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Compressionratio')
    plt.ylabel('psnr')
    plt.legend()

    filepath = 'plots/' + 'test_volume_' + 'SmallifyDifferentNWSizes_4_Hypersearch_For_Each_Compression' + '.png'
    plt.savefig(filepath)


def ResnetVSNoResnet():
    BASENAMENoResnetPruned = 'experiments/diff_comp_rates/test_experiment_NoResNetSiren/4C/test_experiment_'

    experimentNames = ['50_0', '50_1', '50_2', '100_0', '100_1', '100_2', '200_0', '200_1', '200_2']

    BASENAMENoResnetUnPruned = 'experiments/diff_comp_rates/test_experiment_NoResNetSiren/4C_NoPrune/test_experiment'
    experimentNamesUnPruned = ['100_0', '100_1', '100_2', '200_0', '200_1', '200_2']

    BASENAMEResnetPruned = 'experiments/diff_comp_rates/test_experiment_BroaderNW/4C/test_experiment_'

    BASENAMEResnetUnPruned = 'experiments/diff_comp_rates/test_experiment_BroaderNW/4C_NoPrune/test_experiment'

    InfoName = 'info.txt'

    compressionRatioNoResnetPruned = []
    PSNRNoResnetPruned = []

    compressionRatioNoResnetUnPruned = []
    PSNRNoResnetUnPruned = []

    compressionRatioResnetPruned = []
    PSNRResnetPruned = []

    compressionRatioResnetUnPruned = []
    PSNRResnetUnPruned = []


    # M: generate lists...
    generate_plot_lists(([compressionRatioNoResnetPruned, PSNRNoResnetPruned],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMENoResnetPruned, (InfoName,), experiment_names=experimentNames)

    generate_plot_lists(([compressionRatioNoResnetUnPruned, PSNRNoResnetUnPruned],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMENoResnetUnPruned, (InfoName,), experiment_names=experimentNamesUnPruned)

    generate_plot_lists(([compressionRatioResnetPruned, PSNRResnetPruned],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEResnetPruned, (InfoName,), experiment_names=experimentNames)

    generate_plot_lists(([compressionRatioResnetUnPruned, PSNRResnetUnPruned],),
                        (['compression_ratio', 'psnr'],),
                        BASENAMEResnetUnPruned, (InfoName,), experiment_names=experimentNames)

    # M: generate avg values

    compressionRatioNoResnetPruned = generateMeanValues(compressionRatioNoResnetPruned, 3)
    PSNRNoResnetPruned = generateMeanValues(PSNRNoResnetPruned, 3)

    compressionRatioNoResnetUnPruned = generateMeanValues(compressionRatioNoResnetUnPruned, 3)
    PSNRNoResnetUnPruned = generateMeanValues(PSNRNoResnetUnPruned, 3)

    compressionRatioResnetPruned = generateMeanValues(compressionRatioResnetPruned, 3)
    PSNRResnetPruned = generateMeanValues(PSNRResnetPruned, 3)

    compressionRatioResnetUnPruned = generateMeanValues(compressionRatioResnetUnPruned, 3)
    PSNRResnetUnPruned = generateMeanValues(PSNRResnetUnPruned, 3)

    # M: plot
    plt.plot(compressionRatioNoResnetPruned, PSNRNoResnetPruned, label='No Resnet, Pruned')
    plt.plot(compressionRatioNoResnetUnPruned, PSNRNoResnetUnPruned, label='No Resnet, not Pruned', alpha=0.6)

    plt.plot(compressionRatioResnetPruned, PSNRResnetPruned, label='Use Resnet, Pruned')
    plt.plot(compressionRatioResnetUnPruned, PSNRResnetUnPruned, label='Use Resnet, not Pruned', alpha=0.6)

    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Compressionratio')
    plt.ylabel('psnr')
    plt.legend()

    filepath = 'plots/' + 'test_volume_' + 'ResnetVSNoResnet_4Channel' + '.png'
    plt.savefig(filepath)


if __name__ == '__main__':
    #rmseTTHRESHExperiment()
    #paramExperiment()
    #QuantBitsExperiment()
    #QuantVsOrigExperiment()
    #OrigVSSelfImplmentation()
    #NumberOfChannelsVSCompression()
    #QuantbitsVSCompressionratio()
    #CompressionGainVSPSNR()
    #PrunedVSUnpruned()
    #influenceSmallifyParameter()
    #influenceHyperSearchParams()
    #SmallifyVSNeurcompLR()
    #SmallifyAftertrainingVSNoAftertrainig()
    SmallifyDifferentNWSizes()
    #ResnetVSNoResnet()
