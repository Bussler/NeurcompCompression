from NeurcompTraining import config_parser
import training.hyperparameter_search as hypersearch


if __name__ == '__main__':
    parser = config_parser()
    args = vars(parser.parse_args())

    random_search_spaces = {
        #"grad_lambda": ([0.005, 0.5], 'log'),
        #"grad_lambda": [0.00005, 0.00001, 0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.5, 0.1, 1],
        #"lr": [5e-6, 5e-5, 5e-4, 5e-3],
        #"grad_lambda": [0.01, 0.001, 0.0001],
        #"lr": [0.001, 0.0001, 0.00001],
        "lr": ([0.0001, 0.0007]),
        "grad_lambda": ([1e-04, 1e-05, 1e-06]),
        "lambda_betas": ([3e-04, 3e-05, 3e-06]),
        "lambda_weights": ([1e-06, 1e-05, 1e-07]),
        #"pruning_threshold": ([0.7, 0.95], 'float')
    }

    #results = hypersearch.random_search(args, random_search_spaces, num_search=20)
    results = hypersearch.grid_search(args, random_search_spaces)
    print('All Results: ', results)
