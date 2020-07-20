import pandas as pd
from datetime import timedelta
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import DateFormatter
import numpy as np
from os.path import join as join_path
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesRegressor

# START_DATE = '2014-09-29 00:00:00'
START_DATE = '2017-03-09 00:00:00'
END_DATE = '2017-05-29 23:00:00'
PATH = 'dataset'
NUMERAL = join_path(PATH, 'final_numeral.csv')
COUNT = join_path(PATH, 'count.csv')

sns.set()
# sns.set_palette('coolwarm')

# reading_count = pd.read_csv(COUNT, index_col=0, parse_dates=[0])
# # reading_count = reading_count.resample('AS').sum()

# mask = (reading_count.index >= START_DATE) & \
#        (reading_count.index <= END_DATE)
# reading_count = reading_count.loc[mask]
# # print(reading_count)


# fig, ax = plt.subplots(figsize=(9, 3))
# fig.subplots_adjust(left=.07, bottom=.38, right=1, top=1)
# plt.xlim(reading_count.first_valid_index(), reading_count.last_valid_index())
# ax.xaxis.set_major_formatter(DateFormatter('%d/%m/%Y'))
# # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
# HOUSES = (
#     3,      # 20451
#     13,     # 20468
#     20,     # 20451
# )
# print(reading_count)
# print(reading_count.sum(axis=0))

# for house in HOUSES:
#     sns.lineplot(x=reading_count.index, y=f'kWh_{house}', data=reading_count,
#                  legend='brief', label=f'Casa {house}')
# ax.legend(loc=4)
# ax.tick_params(axis='x', rotation=45)
# ax.set_ylabel('# de Registros')
# ax.set_xlabel('Data (dia)')
# fig.savefig('registro_dia.pdf', dpi=600, format='pdf')
# # plt.show()
# exit()

# exit()

# fig, ax = plt.subplots(figsize=(9, 3))
# fig.subplots_adjust(left=.07, bottom=.28, right=.96, top=1)
# plt.xlim(reading_count.first_valid_index(), reading_count.last_valid_index())
# ax.xaxis.set_major_formatter(DateFormatter('%m/%Y'))
# HOUSES = (
#     # 3, 4, 5, 6, 7,
#     8,                  # 20179
#     # 9, 10, 11, 12,
#     13,                 # 20468
#     # 14,
#     18,                 # 20232
#     # 19, 20,
# )
# print(reading_count.sum(axis=0))
# for house in HOUSES:
#     sns.lineplot(x=reading_count.index, y=f'kWh_{house}', data=reading_count,
#                  legend='brief', label=f'Casa {house}')
# ax.legend(loc=4)
# ax.tick_params(axis='x', rotation=45)
# ax.set_ylabel('# de Registros')
# ax.set_xlabel('Data (mês)')
# fig.savefig('registro_mes.pdf', dpi=600, format='pdf')
# # plt.show()
# exit()


fig, ax = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(left=.12, bottom=.18, right=1.05, top=1)

dataset = pd.read_csv(NUMERAL, index_col=0, parse_dates=[0])
# temperature, weather, weekend, year, day_of_week
dataset.drop(columns=['dst'], inplace=True)
X = dataset.drop(columns=['kWh'])
y = dataset.kWh

# === data correlation ===
corr = dataset.corr('kendall') * 100
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# f, ax = plt.subplots()
sns.heatmap(corr, cmap='coolwarm', center=0, mask=mask, annot=True,
            square=False, linewidths=1, vmax=100, vmin=-100, fmt='.2f',
            cbar_kws={"shrink": .75, 'format': '%.0f%%'})
ax.tick_params(axis='x', rotation=45)
# plt.show()
fig.savefig('correlacao.pdf', dpi=600, format='pdf')

exit()

# === SelectKBest ===
best_features = SelectKBest(score_func=f_regression, k='all')
fitted = best_features.fit(X, y)

scores_ = pd.DataFrame(fitted.scores_)
pvalues_ = pd.DataFrame(fitted.pvalues_)
columns_ = pd.DataFrame(X.columns)
final = pd.concat([columns_, scores_, pvalues_], axis=1)
final.columns = ['Feature', 'Score', 'p-Value']
final['p-Value'] *= 100
print(final.nlargest(13, 'Score'), '\n')
# print(final)

exit()

# # === model importance ===
# model = ExtraTreesRegressor()
# fitted_model = model.fit(X, y)
# print(fitted_model.feature_importances_)
# feat_importances = pd.Series(
#     fitted_model.feature_importances_, index=X.columns)
# feat_importances.nsmallest(12).plot(kind='barh')
# plt.show()

# exit()

data = pd.read_csv('hue/Residential_1.csv')

data.date = pd.to_datetime(data.date)
# data = data[['date', 'energy_kWh']]
data.hour = data.hour.apply(lambda hour: timedelta(hours=hour))
data['datetime'] = data.date + data.hour
data.index = data.datetime
# data = data[['datetime', 'energy_kWh']]
monthly = data.groupby(
    pd.Grouper(freq='M')).energy_kWh.agg(['mean', np.median]).reset_index()
weekly = data.groupby(
    pd.Grouper(freq='W')).energy_kWh.sum().reset_index()

# plt.figure()
# plt.subplot(121)
# sns.lineplot(x=weekly.datetime, y=weekly.energy_kWh).set(
#     xlabel='Weekly', ylabel='Energy (kWh)')

# plt.subplot(122)
# sns.lineplot(x='datetime', y='value', hue='variable',
#              data=monthly.melt(['datetime'])) \
#              .set(xlabel='Monthly', ylabel='Energy (kWh)')

plt.figure()
plt.subplot(211)
# Kernel density estimation
sns.distplot(data.energy_kWh,
             kde=False,
             hist_kws={'edgecolor': 'black'})
plt.subplot(212)
sns.distplot(data.energy_kWh,
             hist=False, rug=True)
plt.show()
