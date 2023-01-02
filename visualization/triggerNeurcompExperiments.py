from training.training import training
from NeurcompTraining import config_parser
from NeurcompQuantization import quantize
from NeurcompDequantization import dequantize
from model.model_utils import write_dict
import os
import visualization.pltUtils as pu
import numpy as np


def neurcompRunsDiffComprRatesFromFrontier():
    parser = config_parser()
    args = vars(parser.parse_args())

    BASENAME = 'experiments/hyperparam_search/mhd_p_NAS/100/mhd_p_100_'
    experimentNames = np.linspace(0, 49, 50, dtype=int)

    InfoName = 'info.txt'
    configName = 'config.txt'

    PSNR = []
    CompressionRatio = []

    pu.generate_plot_lists(([PSNR, CompressionRatio],),
                        (['psnr', 'compression_ratio'],),
                        BASENAME, (InfoName,), experiment_names=experimentNames)

    configs = pu.findParetoValues(CompressionRatio, PSNR)

    BASEEXPNAME = args['expname']

    #for compr in [210,225,235,244,296,388,463,546,596,770,931,1251]:#[50.0, 100.0, 200.0, 300.0, 400.0]:
    for c in configs:

        args['compression_ratio'] = c[0]
        args['lr'] = c[1]
        args['grad_lambda'] = c[2]
        args['n_layers'] = c[3]

        args['expname'] = BASEEXPNAME + str(int(c[0]))
        args['checkpoint_path'] = ''
        args['feature_list'] = None
        training(args)


def neurcompRunsDiffComprRates():
    parser = config_parser()
    args = vars(parser.parse_args())

    BASEEXPNAME = args['expname']

    #for compr in [210,225,235,244,296,388,463,546,596,770,931,1251]:#[50.0, 100.0, 200.0, 300.0, 400.0]:
    for compr in [105, 194, 283, 303, 311, 371, 468, 511, 603, 715, 808, 945, 1354]:

        args['compression_ratio'] = compr
        args['expname'] = BASEEXPNAME + str(int(compr))
        args['checkpoint_path'] = ''
        args['feature_list'] = None
        training(args)


def neurcompRunsVariational():
    parser = config_parser()
    args = vars(parser.parse_args())

    BASEEXPNAME = args['expname']
    BASELOGDIR = args['Tensorboard_log_dir']

    #for run in [1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7]:
    for run in [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    #for run in [1e-7,5e-7,1e-8,5e-8]:
        #args['variational_dkl_multiplier'] = run
        args['variational_lambda_weight'] = run
        args['expname'] = BASEEXPNAME + str(int(run))
        args['Tensorboard_log_dir'] = BASELOGDIR + str(run) + '/0'
        args['checkpoint_path'] = ''
        args['feature_list'] = None
        training(args)


def quantize_dequantize(args, quant_bits, BASENAME, VARIATION_NAME, QUANTNAME):
    args['quant_bits'] = quant_bits
    args['filename'] = QUANTNAME
    quantize(args)

    deqantArgs = {}
    deqantArgs['compressed_file'] = BASENAME + VARIATION_NAME + '/' + QUANTNAME
    deqantArgs['volume'] = args['data']
    deqantArgs['decompressed_file'] = 'testDecompressed'
    info = dequantize(deqantArgs)
    return info


def Do_QuantizeDequantize_shifted():
    #BASENAME = 'experiments/test_experiment_QuantbitsVSCompressionratio/Ratio200/test_experiment'
    BASENAME = 'experiments/diff_comp_rates/mhd_p_QuantbitsVSCompressionratio/Ratio50/mhd_p_50'
    CONFIGNAME = 'config.txt'
    QUANTNAME = 'modelQuant'

    quants = [9,8,7,6,5,4,3,2,1]

    for i, compr in enumerate([50, 43, 36, 31, 26, 21]):
        config_name = BASENAME + str(compr) + '/' + CONFIGNAME
        args = pu.dict_from_file(config_name)
        info = quantize_dequantize(args, quants[i], BASENAME, str(compr), QUANTNAME)


def Do_QuantizeDequantize():
    BASENAME = 'experiments/diff_comp_rates/test_experiment_BroaderNW/2C/test_experiment_'
    CONFIGNAME = 'config.txt'
    QUANTNAME = 'modelQuant'

    results = {}

    experimentNames = [50,100,150,200,300]
    experimentNames2 = ['50_0', '50_1', '50_2', '100_0', '100_1', '100_2', '200_0', '200_1', '200_2', '300_0', '300_1',
                        '300_2', '400_0', '400_1', '400_2']
    #for i in range(0,54):
    #    experimentNames.append(i)

    for compr in experimentNames2:

        config_name = BASENAME + str(compr) + '/' + CONFIGNAME
        args = pu.dict_from_file(config_name)

        for b in [8]: #[3, 5, 7, 9, 10]
            info = quantize_dequantize(args, b, BASENAME, str(compr), QUANTNAME)
            key = 'Compr_'+str(compr)+'_Bit_'+str(b)
            results[key] = info

    print(results)
    write_dict(results, "QuantizeDequantizeDiffBits_Results.txt")


if __name__ == '__main__':
    neurcompRunsDiffComprRates()
    #neurcompRunsDiffComprRatesFromFrontier()
    #neurcompRunsVariational()
    #Do_QuantizeDequantize()
    #Do_QuantizeDequantize_shifted()

