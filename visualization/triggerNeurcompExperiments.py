from training.training import training
from NeurcompTraining import config_parser
from NeurcompQuantization import quantize
from NeurcompDequantization import dequantize
from model.model_utils import write_dict
import os
import visualization.pltUtils as pu


def neurcompRunsDiffComprRates():
    parser = config_parser()
    args = vars(parser.parse_args())

    BASEEXPNAME = args['expname']

    for compr in [50.0, 100.0, 200.0, 300.0, 400.0]:#[20.0, 50.0, 100.0, 200.0, 400.0]:

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

    for run in [0,1,2]:
        args['expname'] = BASEEXPNAME + str(int(run))
        args['Tensorboard_log_dir'] = BASELOGDIR + str(run)
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
    #neurcompRunsDiffComprRates()
    neurcompRunsVariational()
    #Do_QuantizeDequantize()
    #Do_QuantizeDequantize_shifted()

