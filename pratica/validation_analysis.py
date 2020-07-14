import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
import os

PATH = 'predictions'
METRICS = os.path.join(PATH, 'metrics.csv')
MODELS = os.path.join(PATH, 'models')
REAL = os.path.join(MODELS, 'real.csv')


def get_points(dataset, points, head):
    if head:
        return dataset.iloc[:points]
    return dataset.iloc[-points:]


def plot(predictions, y_test, points, head, separate):

    y_test_points = get_points(y_test, points, head)

    if separate:
        fig, ax = plt.subplots(len(predictions))
    else:
        fig, ax = plt.subplots(1)
        ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()

    if separate:
        position = 0
        for name, prediction in predictions:
            prediction_points = get_points(prediction, points, head)
            ax[position].plot(y_test_points, label='Measured')
            ax[position].plot(prediction_points, label=name)
            ax[position].legend()
            position += 1
    else:
        ax.plot(y_test_points, label='Measured')
        for name, prediction in predictions:
            prediction_points = get_points(prediction, points, head)
            ax.plot(prediction_points, label=name)
        ax.legend()

    plt.show()


# TODO
# Data analysis
# Models metrics
# Compare models https://scikit-learn.org/stable/modules/cross_validation.html
# SVR is not scale invariant, so it is highly recommended to scale the data

real = pd.read_csv(REAL, index_col=0, parse_dates=[0])

# NOTE
# EVS, r2                           %   higher is better    0 to 1
# MAPE                              %   lower is better     0 to 1
# MSE, MPD, MGD, ME, MAE, MedAE         lower is better
metrics = pd.read_csv(METRICS, index_col=0)
metrics.drop('real', inplace=True)

dont_plot = [
    # generic
    # 'DummyRegressor=median',
    # 'KNeighborsRegressor=7',

    # tree
    'ExtraTreeRegressor',
    'DecisionTreeRegressor',  # a de baixo eh melhor
    # 'ExtraTreesRegressor',
    'RandomForestRegressor',  # a de cima eh melhor

    # linear good
    'NuSVR=linear',
    'MLPRegressor=1',
    'RANSACRegressor',
    'PassiveAggressiveRegressor',

    # linear bad
    'KernelRidge',
    'RidgeCV',
    'LinearRegression',  # parece igual o de baixo
    'Ridge',             # parece igual o de cima
    'LarsCV',
    'BayesianRidge',
]
predictions = []
for name in metrics.index:
    if name in dont_plot:
        continue
    model_path = os.path.join(MODELS, f'{name}.csv')
    prediction = pd.read_csv(model_path, index_col=0, parse_dates=[0])
    predictions.append((name, prediction))

plot(predictions, real, points=22, head=False, separate=False)
