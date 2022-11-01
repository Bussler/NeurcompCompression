from NeurcompTraining import config_parser
import training.hyperparameter_search as hypersearch


if __name__ == '__main__':
    parser = config_parser()
    args = vars(parser.parse_args())

    results = hypersearch.random_search(args)
    print('All Results: ', results)
