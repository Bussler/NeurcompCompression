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
    