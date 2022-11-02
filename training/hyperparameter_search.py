import random
from math import log10
from training.training import training
from itertools import product


ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def random_search_spaces_to_config(random_search_spaces):
    """"
    Takes search spaces for random search as input; samples accordingly
    from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver & network
    """

    config = {}

    for key, (rng, mode) in random_search_spaces.items():
        if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
            print("'{}' is not a valid random sampling mode. "
                  "Ignoring hyper-param '{}'".format(mode, key))
        elif mode == "log":
            if rng[0] <= 0 or rng[-1] <= 0:
                print("Invalid value encountered for logarithmic sampling "
                      "of '{}'. Ignoring this hyper param.".format(key))
                continue
            sample = random.uniform(log10(rng[0]), log10(rng[-1]))
            config[key] = 10 ** (sample)
        elif mode == "int":
            config[key] = random.randint(rng[0], rng[-1])
        elif mode == "float":
            config[key] = random.uniform(rng[0], rng[-1])
        elif mode == "item":
            config[key] = random.choice(rng)

    return config


def find_best_config(configs, args):

    best_val = None
    best_config = None
    best_name = None
    results = []

    baseExpName = args['expname']

    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format(
            (i + 1), len(configs)), configs[i])

        for key in configs[i]:
            args[key] = configs[i][key]

        args['expname'] = baseExpName + str(i)
        args['checkpoint_path'] = ''

        info = training(args, verbose=False)
        results.append(info)

        if not best_val or info['rmse'] < best_val:
            best_val, best_config, best_name = info['rmse'], configs[i], args['expname']

    print("\nSearch done. Best Val Loss = {}".format(best_val))
    print("Best Config:", best_config, ' Name of config: ', best_name)
    return list(zip(configs, results))


def grid_search(args, grid_search_spaces = {
                    "grad_lambda": [0.015, 0.05],
                    "lambda_betas": [5e-3, 5e-4, 5e-5, 5e-6],
                    "lambda_weights": [5e-5, 5e-6, 5e-7],
                }):

    configs = []
    for instance in product(*grid_search_spaces.values()):
        configs.append(dict(zip(grid_search_spaces.keys(), instance)))

    return find_best_config(configs, args)


def random_search(args, random_search_spaces = {
                      "grad_lambda": ([0.0005, 0.5], 'log'),
                      "lambda_betas": ([0.000005, 0.5], 'log'),
                      "lambda_weights": ([0.0000005, 0.05], 'log'),
                  }, num_search=20):

    configs = []
    for _ in range(num_search):
        configs.append(random_search_spaces_to_config(random_search_spaces))

    return find_best_config(configs, args)

