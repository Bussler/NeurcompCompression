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

    parser.add_argument('--d_in', type=int, default=3, help='spatial input dimension')
    parser.add_argument('--d_out', type=int, default=1, help='spatial output dimension')

    parser.add_argument('--grad_lambda', type=float, default=0.05,
                        help='lambda weighting term for gradient regularization - if 0, no regularization is performed; default=0.05')

    parser.add_argument('--n_layers', type=int, default=8, help='number of layers for the network')
    parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint from where to load model')

    parser.add_argument('--omega_0', default=30, help='scale for SIREN')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=5e-5')
    parser.add_argument('--max_pass', type=int, default=75,
                        help='number of training passes to make over the volume, default=75')
    parser.add_argument('--pass_decay', type=int, default=20,
                        help='training-pass-amount at which to decay learning rate, default=20')
    parser.add_argument('--lr_decay', type=float, default=.2, help='learning rate decay, default=.2')

    parser.add_argument('--compression_ratio', type=float, default=50,
                        help='the data is compressed by #voxels / compression_ratio')

    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--sample_size', type=int, default=16, help='how many indices to generate per batch item')
    parser.add_argument('--num_workers', type=int, default=8, help='how many parallel workers for batch access')

    # M: enable option for quantization, various dropout methods
    parser.add_argument('--dropout_technique', type=str, default='', help='Dropout technique to prune the network')
    parser.add_argument('--lambda_betas', type=float, default=0.05, help='lambda weighting term for dropout weights')
    parser.add_argument('--lambda_weights', type=float, default=0.05, help='lambda scaling term for nw weights in case of dropout')

    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = vars(parser.parse_args())
    print("Finished parsing arguments, starting training")
    training(args)

