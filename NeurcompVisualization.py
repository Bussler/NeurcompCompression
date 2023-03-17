from NeurcompTraining import config_parser
from visualization.OutputToVTK import tiled_net_out
from model.model_utils import setup_neurcomp
from data.IndexDataset import get_tensor, IndexDataset

if __name__ == '__main__':
    parser = config_parser()

    args = vars(parser.parse_args())
    print("Finished parsing arguments, starting visualizing model.")

    volume = get_tensor(args['data'])
    dataset = IndexDataset(volume, args['sample_size'])

    model = setup_neurcomp(args['compression_ratio'], dataset.n_voxels, args['n_layers'], args['d_in'],
                           args['d_out'], args['omega_0'], args['checkpoint_path'], '',
                           args['pruning_momentum'], use_resnet=args['use_resnet'], featureList=args['feature_list'],
                           pruning_threshold= args['pruning_threshold'],
                           variational_init_droprate=args['variational_init_droprate'])

    tiled_net_out(dataset, model, True, gt_vol=volume, evaluate=True, probalistic_model=False,
                  write_vols=True)

    print("Finished visualization.")
