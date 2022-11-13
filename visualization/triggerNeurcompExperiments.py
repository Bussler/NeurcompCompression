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

    for compr in [50, 43.09, 36.99, 31.49, 26.39, 21.09, 15.99, 10.69, 5.49]:#[20, 50, 100, 200, 400]:
        args['compression_ratio'] = compr
        args['expname'] = BASEEXPNAME + str(int(compr))
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
    BASENAME = 'experiments/hyperparam_search/test_experiment_smallify_RandomSearch/test_experimentHyperSearch'
    CONFIGNAME = 'config.txt'
    QUANTNAME = 'modelQuant'

    results = {}

    experimentNames = []
    for i in range(0,20):
        experimentNames.append(i)

    for compr in experimentNames:

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
    Do_QuantizeDequantize()
    #Do_QuantizeDequantize_shifted()

