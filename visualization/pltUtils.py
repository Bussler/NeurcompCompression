from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def generate_array_MLFlow(client, idList, searchedParam):

    data = []
    realCompRates = []

    for id in idList:
        runs = client.search_runs(experiment_ids=id)
        bestrun = None
        for entry in runs:
            if bestrun is None or entry.data.params[searchedParam] < bestrun.data.params[searchedParam]:
                bestrun = entry
        data.append(float(bestrun.data.params[searchedParam]))
        realCompRates.append(float(bestrun.data.params['compression_ratio']))

    return data, realCompRates


def dict_from_file(filename):
    file = open(filename, 'r')
    Lines = file.readlines()

    d = {}
    for line in Lines:
        line = line.replace(' ', '')
        line = line.replace('\n', '')
        lineParts = line.split('=')

        value = lineParts[1]

        # M: parse int, float, list or string
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if ',' in value:
                    value = value.replace('[', '')
                    value = value.replace(']', '')
                    value = value.split(',')

                    try:
                        value = [int(x) for x in value]
                    except ValueError:
                        value = [float(x) for x in value]
                else:
                    value = lineParts[1]

        d[lineParts[0]] = value

    return d


def append_lists_from_dicts(lists: [], dict: {}, keys: []):
    if len(lists) is not len(keys):
        print("Error filling lists")

    for i, key in enumerate(keys):
        lists[i].append(dict[key])


def generate_plot_lists(lists: ([]), keys: ([]), BASENAME, config_names: (), experiment_names: []):

    if len(lists) is not len(keys) or len(lists) is not len(config_names):
        print("Error filling lists")

    for compr in experiment_names:
        for i in range(len(lists)):
            config_name = BASENAME + str(compr) + '/' + config_names[i]
            info = dict_from_file(config_name)
            append_lists_from_dicts(lists[i], info, keys[i])


def normalize_array_0_1(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalize_array(array, minV, maxV, minN, maxN):
    return (maxN-minN) * ((array - minV) / (maxV - minV)) + minN