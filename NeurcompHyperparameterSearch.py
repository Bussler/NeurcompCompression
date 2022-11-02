from NeurcompTraining import config_parser
import training.hyperparameter_search as hypersearch


if __name__ == '__main__':
    parser = config_parser()
    args = vars(parser.parse_args())

    random_search_spaces = {
        #"grad_lambda": ([0.005, 0.5], 'log'),
        #"lambda_betas": ([0.000005, 0.005], 'log'),
        #"lambda_weights": ([0.0000005, 0.0005], 'log'),
        "pruning_threshold": ([0.1, 0.7], 'float')
    }

    results = hypersearch.random_search(args, random_search_spaces, num_search=10)
    #results = hypersearch.grid_search(args)
    print('All Results: ', results)
