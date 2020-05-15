from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from scipy import stats

iris_dataset = load_iris()
iris_data = iris_dataset['data']                    # objetos
iris_features = iris_dataset['feature_names']       # atributos
iris_features = [name[:-5] for name in iris_features]
iris_targets = iris_dataset['target']               # atributos / classes
iris_target_names = iris_dataset['target_names']    # classes

data = pd.DataFrame(iris_data, columns=iris_features)
data['target'] = iris_targets
data['target'] = data['target'].apply(lambda x: iris_target_names[x])


# <-- univariados -->
# <-- localizacao ou tendencia central -->

print('Frequency')

# https://numpy.org/doc/stable/reference/generated/numpy.bincount.html
# you can also give weights for each target
frequency = np.bincount(iris_targets)
print('np.bincount', frequency)

# https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
# similar to bincount, but you may also get the frequency in percentage
hist, bin_edges = np.histogram(iris_targets, bins=3)
hist_perc, bin_edges_perc = np.histogram(iris_targets, bins=3, density=True)
print('np.histogram', hist)
print('np.histogram', hist_perc, hist_perc * np.diff(bin_edges_perc))

# https://numpy.org/doc/stable/reference/generated/numpy.unique.html
# similar to bincount, but you may also use a string array
# good to parse a string array into an int array
# may also return frequency
targets, index_1st_occurrence, target_array, target_frequency = np.unique(
    data['target'], return_index=True, return_inverse=True, return_counts=True)
print(
    'np.unique', targets, index_1st_occurrence, target_array, target_frequency)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.relfreq.html
# similar to bincount, but return the frequency in percentage instead
frequency, lower_limit, binsize, extra_points = stats.relfreq(
    iris_targets, numbins=3)
print('sp.stats.relfreq', frequency, lower_limit, binsize, extra_points)

print('\nMean, Median, Mode, Quantile, Percentile')
mean = np.mean(data['sepal length'])
print('np.mean', mean)

trimmed_mean = stats.trim_mean(data['sepal length'], .25)
print('sp.stats.trim_mean (25%)', trimmed_mean)

median = np.median(data['sepal length'])
print('np.median', median)

quantile = np.quantile(data['sepal length'], .25)
print('np.quantile 1st (25%)', quantile)

quantile = np.quantile(data['sepal length'], .5)
print('np.quantile 2st (50%, meadian)', quantile)

percentile = np.percentile(data['sepal length'], 25)
print('np.percentile (25%, 1st quantile)', percentile)

percentile = np.percentile(data['sepal length'], 50)
print('np.percentile (50%, 2st quantile, meadian)', percentile)

# mode is the same as meadian, but for symbolic data
mode, count = stats.mode(data['target'])
print('sp.stats.mode', mode, count)


# <-- univariados -->
# <-- dispersao ou espelhamento -->


# <-- multivariados -->

# https://docs.scipy.org/doc/scipy/reference/stats.html
# https://medium.com/@harimittapalli/exploratory-data-analysis-iris-dataset-9920ea439a3e
# univariados
#   dispersao ou espelhamento
#       intervalo
#       variancia
#       desvio padrao
#       outros
#           desvio medio absoluto (DMA)
#           desvio mediano absoluto (DMedA)
#           intervalo inter-quartis (IQ)
#   distribuição ou formato
#       momento
#           k=3 ou obliquidade (skewness)
#           k=4 ou curtose (kurtosis)
#       histograma
#       grafico de pizza
# multivariados
#   dispersao ou espalhamento
#       matriz de covariancia
#       matriz de correlacao
#   gráficos
#       scatter plot
#       bag plot
#       faces de chernoff
#       star plots
#       heatmaps
