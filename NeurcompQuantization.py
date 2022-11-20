from NeurcompTraining import config_parser
from model.model_utils import setup_neurcomp
from data.IndexDataset import get_tensor, IndexDataset
from quantization.Net_Quantizer import NetEncoder
import os


def quantize(args):
    # M: Setup model
    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])

    model = setup_neurcomp(args['compression_ratio'], dataset.n_voxels, args['n_layers'],
                           args['d_in'], args['d_out'], args['omega_0'], args['checkpoint_path'],
                           dropout_technique=args['dropout_technique']+'_quant',
                           featureList=args['feature_list'],
                           use_resnet=args['use_resnet'])

    # M: Setup encoder
    ExperimentPath = os.path.abspath(os.getcwd()) + args['basedir'] + args['expname'] + '/'
    os.makedirs(ExperimentPath, exist_ok=True)
    filename = os.path.join(ExperimentPath, args['filename'])

    encoder = NetEncoder(model)

    print("Start quantization to ", filename)
    encoder.encode(filename, args['quant_bits'])


if __name__ == '__main__':
    # M: Parse Arguments
    parser = config_parser()

    parser.add_argument('--quant_bits', type=int, default=9,
                        help='number b of bits for k-means clustering. 2^b bits; default b = 9')
    parser.add_argument('--filename', type=str, required=True,
                        help='file for quantization.')
    args = vars(parser.parse_args())

    print("Finished parsing arguments.")

    quantize(args)

    print("Done with quantization.")
