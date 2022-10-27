from data.IndexDataset import get_tensor, IndexDataset
import configargparse
import os
from quantization.Net_Dequantizer import NetDecoder
from visualization.OutputToVTK import tiled_net_out


def dequantize(args):

    decoder = NetDecoder()
    net = decoder.decode(args['compressed_file'])
    net.eval()

    if args['volume'] and args['decompressed_file']:
        # M: Setup volume for comparison
        volume = get_tensor(args['volume'])
        dataset = IndexDataset(volume, 16)

        compressed_size = os.path.getsize(args['compressed_file'])
        compression_ratio = (dataset.n_voxels * 4) / compressed_size  # M: Size in bytes
        print('compression ratio:', compression_ratio)

        psnr, l1_diff, mse, rmse = tiled_net_out(dataset, net, True, gt_vol=volume.cpu(), evaluate=True,
                                                 write_vols=True, filename=args['decompressed_file'])

        info = {}
        info['Quant_Compression_Ratio'] = compression_ratio
        info['psnr'] = psnr
        info['l1_diff'] = l1_diff
        info['mse'] = mse
        info['rmse'] = rmse

        ExperimentPath = os.path.dirname(args['compressed_file']) + '/'
        os.makedirs(ExperimentPath, exist_ok=True)

        def write_dict(dictionary, filename):
            with open(os.path.join(ExperimentPath, filename), 'w') as f:
                for key, value in dictionary.items():
                    f.write('%s = %s\n' % (key, value))

        write_dict(info, 'Dequant_Info.txt')
        print('Writing Info into: ', ExperimentPath)


if __name__ == '__main__':
    # M: Parse Arguments
    parser = configargparse.ArgumentParser()
    parser.add_argument('--compressed_file', type=str, required=True,
                        help='path to compressed file')
    parser.add_argument('--volume', type=str, help='path to volumetric dataset')
    parser.add_argument('--decompressed_file', type=str, help='path to decompressed file')
    args = vars(parser.parse_args())

    # M: Setup Decoder
    print("Finished parsing arguments, let's decompress")

    dequantize(args)

    print("Done.")
