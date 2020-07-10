from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import explained_variance_score, max_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def get_name(Model, **kwargs):
    try:
        return f'{Model.__name__}={next(iter(kwargs.values()))}'
    except StopIteration:
        return f'{Model.__name__}'


def train_and_test(Model, train, test, *args, **kwargs):
    algorithm = Model(*args, **kwargs)
    X_train, y_train = train
    X_test, y_test = test
    regressor = algorithm.fit(X_train, y_train)
    y_predict = regressor.predict(X_test)

    print(get_name(Model, **kwargs))

    mse = mean_squared_error(y_test, y_predict)
    mpd = mean_poisson_deviance(y_test, y_predict)
    mgd = mean_gamma_deviance(y_test, y_predict)

    mae = mean_absolute_error(y_test, y_predict)
    mape = mean_absolute_percentage_error(y_test, y_predict) * 100
    evs = explained_variance_score(y_test, y_predict) * 100
    me = max_error(y_test, y_predict)
    medae = median_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict) * 100
    # msle = mean_squared_log_error(y_test, y_predict)

    print(f'    EVS -> {evs:5.2f}% (higher is better)')
    print(f'     r2 -> {r2:5.2f}% (higher is better)')
    print(f'   MAPE -> {mape:5.2f}% (lower is better)')
    print(f'    MSE -> {mse:5.2f}  (lower is better)')
    print(f'    MPD -> {mpd:5.2f}  (lower is better)')
    print(f'    MGD -> {mgd:5.2f}  (lower is better)')
    print(f' MaxErr -> {me:5.2f}  (lower is better)')
    print(f'    MAE -> {mae:5.2f}  (lower is better)')
    # print(f'   MSLE -> {msle:5.2f}  (lower is better)')
    print(f'  MedAE -> {medae:5.2f}  (lower is better)')
    print('\n')
    return y_predict


def plot(points, head, predictions, y_test, together):

    plt.figure()

    if together:
        if head:
            plt.plot(y_test.iloc[:points].index, y_test.iloc[:points],
                     label='Measured')
        else:
            plt.plot(y_test.iloc[-points:].index, y_test.iloc[-points:],
                     label='Measured')
        for name, prediction in predictions:
            if head:
                plt.plot(y_test.iloc[:points].index, prediction[:points],
                         label=name)
            else:
                plt.plot(y_test.iloc[-points:].index, prediction[-points:],
                         label=name)
        plt.legend()

    else:
        size = len(predictions)
        if size <= 3:
            position = (100 * size) + 10
        elif size == 4:
            position = 220
        elif size > 4:
            position = (-(-size // 2) * 100) + 20

        for name, prediction in predictions:
            position += 1
            plt.subplot(position)
            if head:
                plt.plot(y_test.iloc[:points].index, y_test.iloc[:points],
                         label='Measured')
                plt.plot(y_test.iloc[:points].index, prediction[:points],
                         label=name)
            else:
                plt.plot(y_test.iloc[-points:].index, y_test.iloc[-points:],
                         label='Measured')
                plt.plot(y_test.iloc[-points:].index, prediction[-points:],
                         label=name)
            plt.legend()
    plt.show()


def train_test_and_plot(args, train, test, points, head, together):
    predictions = []
    _, y_test = test
    real_min = y_test.min()
    real_max = y_test.max()
    print(f'Real values range from {real_min:.2f} ~ {real_max:.2f}\n')
    for Model, kwargs in args:
        predictions.append((
            get_name(Model, **kwargs),
            train_and_test(Model, train, test, **kwargs)))

    plot(points, head, predictions, y_test, together)


dataset = pd.read_csv('dataset_numeral.csv', index_col=0, parse_dates=[0])

dataset_test = dataset.tail(len(dataset) // 3)
dataset_train = dataset.drop(dataset_test.index)

X_train = dataset_train.drop(columns='kWh')
y_train = dataset_train.kWh
train = (X_train, y_train)

X_test = dataset_test.drop(columns='kWh')
y_test = dataset_test.kWh
test = (X_test, y_test)


# TODO
# SVR is not scale invariant, so it is highly recommended to scale the data

train_test_and_plot([
    # (KNeighborsRegressor, {'n_neighbors': 7}),
    # (DecisionTreeRegressor, {}),  # a de baixo eh melhor
    (ExtraTreesRegressor, {}),
    # (RandomForestRegressor, {}),  # a de cima eh melhor
    # (NuSVR, {'kernel': 'linear'}),
    # (MLPRegressor, {'random_state': 1}),
    # (KernelRidge, {'alpha': .5}),
    # (DummyRegressor, {'strategy': 'median'}),
], train, test, 500, head=False, together=False)
