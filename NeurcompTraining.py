from training.training import training


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    parser.add_argument("--expname", type=str, required=True,
                        help='name of your experiment; is required')
    parser.add_argument("--data", type=str, required=True,
                        help='path to volume data set; is required')
    parser.add_argument("--basedir", type=str, default='/experiments/',
                        help='where to store ckpts and logs')
    parser.add_argument("--Tensorboard_log_dir", type=str, default='',
                        help='where to store tensorboard logs')

    parser.add_argument('--d_in', type=int, default=3, help='spatial input dimension')
    parser.add_argument('--d_out', type=int, default=1, help='spatial output dimension')

    parser.add_argument('--grad_lambda', type=float, default=0.001,
                        help='lambda weighting term for gradient regularization - if 0, no regularization is performed; default=0.05')

    parser.add_argument('--n_layers', type=int, default=8, help='number of layers for the network')
    parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint from where to load model')
    parser.add_argument('--feature_list', type=int, nargs="+", default=None, help='list of internal features to handle')
    parser.add_argument('--use_resnet', dest='use_resnet', action='store_true', help='Use residual blocks in the model')
    parser.add_argument('--dont_use_resnet', dest='use_resnet', action='store_false',
                        help='Do not use residual blocks in the model')
    parser.set_defaults(use_resnet=True)

    parser.add_argument('--omega_0', default=30, help='scale for SIREN')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate, default=5e-5')
    parser.add_argument('--max_pass', type=int, default=75,
                        help='number of training passes to make over the volume, default=75')
    parser.add_argument('--pass_decay', type=int, default=20,
                        help='training-pass-amount at which to decay learning rate, default=20')
    parser.add_argument('--lr_decay', type=float, default=.2, help='learning rate decay, default=.2')
    parser.add_argument('--smallify_decay', type=int, default=0,
                        help='Option to enable loss decay as presented in the smallify-paper:'
                             'Every smallify_decay - epochs without improvement, the learning rate'
                             'is divided by 10 up until 1e-7. Default: 0 to turn option off')

    parser.add_argument('--compression_ratio', type=float, default=50,
                        help='the data is compressed by #voxels / compression_ratio')

    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--sample_size', type=int, default=16, help='how many indices to generate per batch item')
    parser.add_argument('--num_workers', type=int, default=8, help='how many parallel workers for batch access')

    # M: enable option for quantization, various dropout methods
    parser.add_argument('--dropout_technique', type=str, default='', help='Dropout technique to prune the network')
    parser.add_argument('--lambda_betas', type=float, default=5e-2, help='lambda weighting term for dropout weights')
    parser.add_argument('--lambda_weights', type=float, default=5e-6, help='lambda scaling term for nw weights in case of dropout')
    parser.add_argument('--pruning_momentum', type=float, default=0.02, help='momentum for sign variance')
    parser.add_argument('--pruning_threshold', type=float, default=0.9,
                        help='betas with a higher variance than this will get pruned')

    parser.add_argument('--variational_sigma', type=float, default=0.0,
                        help='sigma for the stochastical nw prediction')
    parser.add_argument('--variational_init_droprate', type=float, default=0.5,
                        help='Droprate to initialize drop layers with')
    parser.add_argument('--variational_dkl_multiplier', type=float, default=1e-8,
                        help='increase dkl during training')
    parser.add_argument('--variational_lambda_dkl', type=float, default=1e-5,
                        help='lambda scaling term for dkl in case of variational dropout')
    parser.add_argument('--variational_lambda_weight', type=float, default=1e-5,
                        help='lambda scaling term for nw weights in case of variational dropout')
    parser.add_argument('--variational_lambda_entropy', type=float, default=1e-5,
                        help='lambda scaling term for dropout entropy in case of variational dropout')

    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = vars(parser.parse_args())
    print("Finished parsing arguments, starting training")
    training(args)

