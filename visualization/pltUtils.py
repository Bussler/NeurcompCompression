from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


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
                    value = [int(x) for x in value]
                else:
                    value = lineParts[1]

        d[lineParts[0]] = value

    return d
