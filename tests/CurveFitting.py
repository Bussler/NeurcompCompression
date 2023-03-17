import numpy as np
import visualization.pltUtils as pu
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tikzplotlib


def get_pareto_data(BASENAME, experimentNames, InfoName='info.txt', configName='config.txt'):

    PSNR = []
    CompressionRatio = []

    pu.generate_plot_lists(([PSNR, CompressionRatio],),
                           (['psnr', 'compression_ratio'],),
                           BASENAME, (InfoName,), experiment_names=experimentNames)

    pareto_front = pu.plot_pareto_frontier(CompressionRatio, PSNR)

    upper_limit = 1000
    pareto_configs = []
    pareto_infos = []

    for ppair in pareto_front:
        c = ppair[0]
        p = ppair[1]
        for eN in experimentNames:
            foldername = BASENAME + str(eN)
            cName = foldername + '/'+InfoName

            info = pu.dict_from_file(cName)
            if info['compression_ratio'] == c and c < upper_limit:
                config = pu.dict_from_file(foldername+'/'+configName)

                config['compression_ratio'] = c
                config['psnr'] = p

                pareto_configs.append(config)
                pareto_infos.append(info)

    return pareto_configs, pareto_infos


def split_data(configs:[], infos:[]):

    paretoCompr = []
    paretoPsnr = []
    paretoLBeta = []
    paretoLWeight = []
    paretoMomentum = []
    paretoThreshold = []
    paretoDKLMult = []
    paretoSigma = []
    paretoLambdaDKL = []
    paretoLambdaWeight = []
    paretoInitDroprate = []

    pareto_PrevCompr = []
    pareto_GridSize = []
    pareto_FeatureSize = []

    for entry, info in zip(configs, infos):
        paretoCompr.append(entry['compression_ratio'])
        paretoPsnr.append(entry['psnr'])
        paretoLBeta.append(entry['lambda_betas'])
        paretoLWeight.append(entry['lambda_weights'])
        paretoMomentum.append(entry['pruning_momentum'])
        paretoThreshold.append(entry['pruning_threshold'])
        paretoLambdaDKL.append(entry['variational_lambda_dkl'])
        paretoDKLMult.append(entry['variational_dkl_multiplier'])
        paretoSigma.append(entry['variational_sigma'])
        paretoLambdaWeight.append(entry['variational_lambda_weight'])
        paretoInitDroprate.append(entry['variational_init_droprate'])

        #elementes_pre_prune = (entry['features_per_layer'] ** 2) * 2 * 1.0 * (entry['n_layers']-1.0)
        #elements_volume = info['volume_num_voxels']
        #pre_prune_compr = elements_volume / elementes_pre_prune
        #pareto_PrevCompr.append(pre_prune_compr)

    return {'compression_ratio' : paretoCompr,
            'psnr' : paretoPsnr,
            'lambda_betas' : paretoLBeta,
            'lambda_weights' : paretoLWeight,
            'pruning_momentum' : paretoMomentum,
            'pruning_threshold': paretoThreshold,
            'variational_dkl_multiplier' : paretoDKLMult,
            'variational_sigma' : paretoSigma,
            'variational_init_droprate' : paretoInitDroprate,
            'variational_lambda_weight' : paretoLambdaWeight,
            'variational_lambda_dkl' : paretoLambdaDKL,
            #'pre_prune_compr': pareto_PrevCompr,
            }


def second_degree_polynomial(x, a, b, c):
    return a * x + b * x ** 2 + c


def second_degree_polynomial_MultiDim(X, a1, a2, b1, b2, c):
    x1, x2 = X
    return a1 * x1 + a2 * x2 + b1 * x1 ** 2 + b2 * x2 ** 2 + c


def simple_exponential(x, a, b):
    return a * np.power(x, b)

def simple_exponential_log(x, a, b):
    return b * x + np.log(a)


def simple_linear(x, a, b):
    return a * x


def Gauss(x, A, B, C):
    y = A*np.exp((((x-B)**2) / (2 * C**2)))
    return y


def third_degree_polynomial(x, a, b, c, d):
    return (a * x) + (b * x**2) + (c * x**3) + d


def fifth_degree_polynomial(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f


def fit_curve():
    # M: Get data
    #BASENAME = 'experiments/hyperparam_search/mhd_p_Variational_NAS/100/mhd_p_100_'
    #experimentNames = np.linspace(0, 39, 40, dtype=int)
    #experimentNames = np.delete(experimentNames, 8, axis=0)
    #experimentNames = np.delete(experimentNames, 8, axis=0)

    #BASENAME = 'experiments/hyperparam_search/mhd_p_Variational_NAS/100_Dynamic_WithEntropy/mhd_p_'
    #experimentNames = np.linspace(0, 49, 50, dtype=int)

    BASENAME = 'experiments/hyperparam_search/mhd_p_Variational_NAS/Features_PerLayer_dynamic_variance/mhd_p_'
    experimentNames = np.linspace(0, 62, 63, dtype=int)

    #BASENAME = 'experiments/hyperparam_search/mhd_p_NAS/200_controlrun/mhd_p_'
    #experimentNames = np.linspace(0, 47, 48, dtype=int)

    #BASENAME = 'experiments/hyperparam_search/mhd_p_NAS/100/mhd_p_100_'
    #experimentNames = np.linspace(0, 49, 50, dtype=int)

    pareto_configs, pareto_infos = get_pareto_data(BASENAME, experimentNames)
    data_list = split_data(pareto_configs, pareto_infos)

    # M: curve fitting for linear model
    x_var = 'compression_ratio'#'psnr'
    #x_var2 = 'lambda_weight_loss'
    y_var = 'variational_dkl_multiplier'  # 'variational_dkl_multiplier'#'lambda_drop_loss'#'variational_sigma' #lambda_betas

    x = np.log(np.asarray(data_list[x_var]))
    #x2 = np.asarray(data_list[x_var2])
    y = np.log(np.asarray(data_list[y_var]))

    #x = np.delete(x, 5, axis=0)
    #x2 = np.delete(x2, 5, axis=0)
    #y = np.delete(y, 5, axis=0)

    #X = (x, x2)

    for entry in zip(x, y):
        print(entry[0], entry[1])

    #popt, _ = curve_fit(simple_linear, x, y)
    popt, _ = curve_fit(simple_exponential_log, x, y)
    #popt, _ = curve_fit(simple_exponential, x, y)
    #popt, _ = curve_fit(second_degree_polynomial_MultiDim, X, y)
    #popt, _ = curve_fit(Gauss, x, y)
    #popt, _ = curve_fit(fifth_degree_polynomial, x, y)

    # M: Summarize Result
    #a, b = popt
    #print('y = %.5f * x + %.5f' % (a, b))
    #a, b = popt
    #print('y = %.5f * x ^ %.5f' % (a, b))
    a, b = popt
    print('y = ', b ,' * x + np.log(', a ,')')
    #a, b, c = popt
    #print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
    #a1, a2, b1, b2, c = popt
    #print(popt)
    #A, B, C = popt
    #print(popt)
    #a, b, c, d, e, f = popt
    #print('y = %.5f * x + %.5f * x^2 + %.5f * x^3 + %.5f * x^4 + %.5f * x^5 + %.5f' % (a, b, c, d, e, f))

    #ax = plt.gca()
    #ax.set(xscale = 'log', yscale = 'log')

    # M: Plotting
    plt.scatter(x, y, label='Baseline')  # M: GT

    x_line = np.linspace(min(x), max(x), 20, dtype=float)
    #x_Line2 = np.linspace(min(x2), max(x2), 20, dtype=float)
    #X_line = (x_line, x_Line2)

    #y_line = simple_linear(x_line, a, b)
    y_line = simple_exponential_log(x_line, a, b)
    #y_line = simple_exponential(x_line, a, b)
    #y_line = second_degree_polynomial(x_line, a, b, c)
    #y_line = second_degree_polynomial_MultiDim(X_line, a1, a2, b1, b2, c)
    #y_line = Gauss(x_line, A, B, C)
    #y_line = fifth_degree_polynomial(x_line, a, b, c, d, e, f)

    plt.plot(x_line, y_line, '--', label='Fitted', color='crimson')

    plt.xlabel('log '+x_var)
    plt.ylabel('log '+y_var)
    plt.legend()

    #filepath = 'plots/LatexFigures/Hyperparam_Analysis/CurveFitting_2/' + 'mhd_p_Variational_Static_thresh_Linear_SetArch'
    filepath = 'plots/test2'
    plt.savefig(filepath + '.pdf')
    plt.savefig(filepath + '.png')
    tikzplotlib.save(filepath + '.pgf')
    pass


def fit_model_complexity_curve():
    BASENAME = 'experiments/hyperparam_search/mhd_p_Variational_NAS/Features_PerLayer_dynamic_variance/mhd_p_'
    experimentNames = np.linspace(0, 62, 63, dtype=int)
    #experimentNames = np.delete(experimentNames, 5, axis=0)
    #experimentNames = np.delete(experimentNames, 5, axis=0)

    #BASENAME = 'experiments/hyperparam_search/mhd_p_Variational_NAS/Features_PerLayer_static_variance_NoEntropy_NoResnet/mhd_p_'
    #experimentNames = np.linspace(0, 59, 60, dtype=int)

    pareto_configs, pareto_infos = get_pareto_data(BASENAME, experimentNames)
    data_list = split_data(pareto_configs, pareto_infos)

    # M: Handling the data
    x_var = 'compression_ratio'
    y_var = 'pre_prune_compr'#'pareto_GridSize'#'pre_prune_compr' # pareto_FeatureSize
    x = np.log(np.asarray(data_list[x_var]))
    y = np.log(np.asarray(data_list[y_var]))

    # M: Linear Regression
    popt, _ = curve_fit(simple_exponential_log, x, y)

    a, b = popt
    print('y = ', b ,' * x + np.log(', a ,')')

    # M: Plotting
    plt.scatter(x, y, label='Baseline')  # M: GT

    x_line = np.linspace(min(x), max(x), 20, dtype=float)
    y_line = simple_exponential_log(x_line, a, b)
    plt.plot(x_line, y_line, '--', label='Fitted', color='crimson')

    plt.xlabel('log '+x_var)
    plt.ylabel('log '+y_var)
    plt.legend()

    filepath = 'plots/LatexFigures/Hyperparam_Analysis/ComprVSPrePruningComplexity/' + 'mhd_p_Variational_Dynamic_SearchArch'
    #filepath = 'plots/test'
    plt.savefig(filepath + '.png')
    plt.savefig(filepath + '.pdf')
    tikzplotlib.save(filepath + '.pgf')

    pass


if __name__ == '__main__':
    fit_curve()
    #fit_model_complexity_curve()
