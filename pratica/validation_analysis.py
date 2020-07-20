import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
from os.path import join as join_path
import seaborn as sns

PATH = 'predictions'
METRICS = join_path(PATH, 'metrics.csv')
MODELS = join_path(PATH, 'models')
REAL = join_path(MODELS, 'real.csv')


def get_points(dataset, points, head):
    if head:
        return dataset.iloc[:points]
    return dataset.iloc[-points:]


def plot(predictions, y_test, points, head, separate):
    y_test_points = get_points(y_test, points, head)

    sns.set()
    # sns.set_palette('coolwarm')

    if separate:
        fig, ax = plt.subplots(len(predictions), figsize=(9, 5))
    else:
        fig, ax = plt.subplots(1, figsize=(9, 5))
        ax.xaxis.set_major_formatter(md.DateFormatter('%d/%m/%Y\n%H:%M'))
        # fig.autofmt_xdate()

    fig.subplots_adjust(left=.08, bottom=.18, right=1, top=1)

    if separate:
        position = 0
        for name, prediction in predictions:
            prediction_points = get_points(prediction, points, head)
            ax[position].plot(y_test_points, label='Real')
            ax[position].plot(prediction_points, label=name)
            ax[position].legend()
            position += 1
    else:
        ax.plot(y_test_points, label='Real')
        for name, prediction in predictions:
            prediction_points = get_points(prediction, points, head)
            ax.plot(prediction_points, label=name)
        ax.legend()

    ax.set_ylabel('kWh')
    ax.set_xlabel('Data')
    ax.legend(loc=2)
    plt.show()
    # fig.savefig('tres_dias_finais_menor.pdf', dpi=600, format='pdf')


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
    # 'MLPRegressor=1',
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
    model_path = join_path(MODELS, f'{name}.csv')
    prediction = pd.read_csv(model_path, index_col=0, parse_dates=[0])
    predictions.append((name, prediction))

plot(predictions, real, points=24 * 3, head=False, separate=False)
