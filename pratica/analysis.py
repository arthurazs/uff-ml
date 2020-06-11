import pandas as pd
from datetime import timedelta
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

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
