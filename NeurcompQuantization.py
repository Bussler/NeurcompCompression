from NeurcompExecution import config_parser
from model.model_utils import setup_neurcomp
from data.IndexDataset import get_tensor, IndexDataset
from quantization.Net_Quantizer import NetEncoder
import os

if __name__ == '__main__':
    parser = config_parser()

    parser.add_argument('--quant_bits', type=int, default=9,
                        help='number b of bits for k-means clustering. 2^b bits; default b = 9')
    parser.add_argument('--filename', type=str, required=True,
                        help='file to quantize / dequantize')
    args = vars(parser.parse_args())

    print("Finished parsing arguments, let's go")

    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])

    model = setup_neurcomp(args['compression_ratio'], dataset.n_voxels, args['n_layers'],
                           args['d_in'], args['d_out'], args['omega_0'], args['checkpoint_path'])

    ExperimentPath = os.path.abspath(os.getcwd()) + args['basedir'] + args['expname'] + '/'
    os.makedirs(ExperimentPath, exist_ok=True)
    filename = os.path.join(ExperimentPath,args['filename'])

    encoder = NetEncoder(model)
    encoder.encode(filename, args['quant_bits'])

    print("Done Quantizing.")


