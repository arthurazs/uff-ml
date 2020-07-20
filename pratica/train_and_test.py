# data science tools
import pandas as pd
import numpy as np
from os.path import join as join_path

# generic models
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor

# tree models
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

# linear models
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV, RANSACRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LarsCV, BayesianRidge
from sklearn.linear_model import PassiveAggressiveRegressor

# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import explained_variance_score, max_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance

PATH = 'predictions'
MODELS = join_path(PATH, 'models')
Y_TEST = join_path(MODELS, 'real.csv')
METRICS = join_path(PATH, 'metrics.csv')
RANGE = join_path(PATH, 'range.txt')
DATASET = join_path('dataset', 'final_numeral.csv')


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def get_name(Model, **kwargs):
    try:
        return f'{Model.__name__}={next(iter(kwargs.values()))}'
    except StopIteration:
        return f'{Model.__name__}'


def train_and_test_one(Model, train, test, *args, **kwargs):
    name = get_name(Model, **kwargs)
    print(f'Training and testing {name}...')

    algorithm = Model(*args, **kwargs)
    X_train, y_train = train
    X_test, y_test = test
    regressor = algorithm.fit(X_train, y_train)
    y_predict = regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_predict)
    mpd = mean_poisson_deviance(y_test, y_predict)
    mgd = mean_gamma_deviance(y_test, y_predict)

    mae = mean_absolute_error(y_test, y_predict)
    mape = mean_absolute_percentage_error(y_test, y_predict)
    evs = explained_variance_score(y_test, y_predict)
    me = max_error(y_test, y_predict)
    medae = median_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    print(f'Saving {name}...\n')
    metrics = pd.DataFrame.from_dict(
        {name: [evs, r2, mape, mse, mpd, mgd, me, mae, medae]}, orient='index')
    metrics.to_csv(METRICS, mode='a', header=False)

    prediction = pd.DataFrame(y_predict, columns=['prediction'])
    prediction.index = X_test.index
    predict_path = join_path(MODELS, f'{name}.csv')
    prediction.to_csv(predict_path)
    return y_predict


def train_and_test_many(args, train, test):
    predictions = []
    _, y_test = test
    real_min = y_test.min()
    real_max = y_test.max()
    print(f'Real values range from {real_min:.2f} ~ {real_max:.2f}\n')
    with open(RANGE, 'w') as range_file:
        range_file.write(f'real min: {real_min}\n')
        range_file.write(f'real max: {real_max}')

    metrics = pd.DataFrame.from_dict(
        {'real': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        orient='index', columns=[
            'EVS', 'r2', 'MAPE', 'MSE', 'MPD', 'MGD', 'ME', 'MAE', 'MedAE'
        ])
    metrics.to_csv(METRICS)

    y_test.to_csv(Y_TEST)

    for Model, kwargs in args:
        predictions.append((
            get_name(Model, **kwargs),
            train_and_test_one(Model, train, test, **kwargs)))


dataset = pd.read_csv(DATASET, index_col=0, parse_dates=[0])

# TODO
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
# https://scikit-learn.org/stable/modules/preprocessing.html
dataset_test = dataset.tail(len(dataset) // 3)
dataset_train = dataset.drop(dataset_test.index)

X_train = dataset_train.drop(columns='kWh')
y_train = dataset_train.kWh
train = (X_train, y_train)

X_test = dataset_test.drop(columns='kWh')
y_test = dataset_test.kWh
test = (X_test, y_test)

train_and_test_many([
    # generic
    (DummyRegressor, {'strategy': 'median'}),
    (KNeighborsRegressor, {'n_neighbors': 7}),

    # tree
    # (ExtraTreeRegressor, {}),
    # (DecisionTreeRegressor, {}),  # a de baixo eh melhor
    (ExtraTreesRegressor, {}),
    # (RandomForestRegressor, {}),  # a de cima eh melhor

    # linear
    # (NuSVR, {'kernel': 'linear'}),
    (MLPRegressor, {'random_state': 1}),
    # (KernelRidge, {}),
    # (RidgeCV, {}),
    # (RANSACRegressor, {}),
    # (LinearRegression, {}),  # parece igual o de baixo
    # (Ridge, {}),  # parece igual o de cima
    # (LarsCV, {}),
    # (BayesianRidge, {}),
    # (PassiveAggressiveRegressor, {}),
], train, test)
