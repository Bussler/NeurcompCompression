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

    BASEEXPNAME = 'Mult'

    for compr in [20, 50, 100, 200, 400]:
        args['compression_ratio'] = compr
        args['expname'] = BASEEXPNAME + str(compr)
        args['checkpoint_path'] = ''
        args['feature_list'] = None
        training(args)


def QuantizeDequantize():
    BASENAME = 'experiments/mhd_p_diffCompRates/mhd_p_'
    CONFIGNAME = 'config.txt'
    QUANTNAME = 'modelQuant'

    results = {}

    for compr in [50, 100, 200, 400]:

        config_name = BASENAME + str(compr) + '/' + CONFIGNAME
        args = pu.dict_from_file(config_name)

        for b in [7]: #[3, 5, 7, 9, 10]
            args['quant_bits'] = b
            args['filename'] = QUANTNAME
            quantize(args)

            deqantArgs = {}
            deqantArgs['compressed_file'] = BASENAME + str(compr) + '/' + QUANTNAME
            deqantArgs['volume'] = args['data']
            deqantArgs['decompressed_file'] = 'testDecompressed'
            info = dequantize(deqantArgs)

            key = 'Compr_'+str(compr)+'_Bit_'+str(b)
            results[key] = info

    print(results)
    write_dict(results, "QuantizeDequantizeDiffBits_Results.txt")


if __name__ == '__main__':
    #neurcompRunsDiffComprRates()
    QuantizeDequantize()

