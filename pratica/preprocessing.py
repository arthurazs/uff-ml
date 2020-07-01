import pandas as pd
from numpy import nan
from datetime import timedelta

START_DATE = '2015-09-29 00:00:00'
END_DATE = '2018-01-29 23:00:00'
END_DATE_P1D = '2018-01-30 23:00:00'  # plus 1 day
HOUSES = (
            # houses located in YVR
            # Vancouver and Lower Mainland area
            3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 19, 20,

            # house located in WYJ
            # Victoria and surrounding area
            # 15,

            # 0 or few data between 2015~2018
            # 1, 17, 26, 27, 28,

            # many missing data between 2015~2018
            # 10, 16, 21, 23, 24, 25,

            # some missing data between 2015~2018
            # 2, 22,

            # almost no missing data between 2015~2018
            # 8, 11, 12, 18,
)


def fix_dst(row):
    if row.name.hour != 1:
        return nan
    return row.dst


def load_dataset(number):
    dataframe = pd.read_csv(f'hue/Residential_{number}.csv', parse_dates=[0])
    dataframe.rename(columns={'energy_kWh': f'kWh_{number}'}, inplace=True)
    dataframe.hour = dataframe.hour.apply(lambda hour: timedelta(hours=hour))
    dataframe['datetime'] = dataframe.date + dataframe.hour
    dataframe.set_index(
        'datetime', inplace=True,
        verify_integrity=False  # TODO Keep duplicate, fix this!
    )
    # isso ta bugando, com integrity false
    # dai to removendo o duplicado, mas isso não é o ideal
    trimmed_df = dataframe[[f'kWh_{number}']]
    # ~ == not
    deduplicated = trimmed_df[~trimmed_df.index.duplicated()]
    # print(trimmed_df.shape, deduplicated.shape)
    return deduplicated


def merge_all(houses):
    left = houses.pop(0)
    while houses:
        right = houses.pop(0)
        left = pd.merge(
            left, right, how='outer',
            left_index=True, right_index=True)
    return left


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis='columns')
    mvt_ren_columns = mis_val_table.rename(columns={0: '#', 1: '%'})
    mvt_ren_columns = mvt_ren_columns[
        mvt_ren_columns.iloc[:, 1] >= 0].sort_values(
            '%', ascending=False).round(1)
    return mvt_ren_columns, df.shape[0]


print('loading weather...')
weather = pd.read_csv('hue/Weather_YVR.csv', parse_dates=[0])
weather.hour = weather.hour.apply(lambda hour: timedelta(hours=hour))
weather['datetime'] = weather.date + weather.hour
weather.set_index('datetime', inplace=True, verify_integrity=True)
weather.drop(columns=['date', 'hour'], inplace=True)
mask = (weather.index >= START_DATE) & (weather.index <= END_DATE)
weather = weather.loc[mask]
weather.weather.fillna(method='bfill', inplace=True)
# print(weather)
# print(weather[weather.weather.isna()])
# print(weather[~weather.weather.isna()])
print('loaded\n')

print('loading holidays...')
# dst -> Daylight Savings Time
holidays = pd.read_csv('hue/Holidays.csv', parse_dates=[0])
holidays.set_index('date', inplace=True, verify_integrity=True)
mask = (holidays.index >= START_DATE) & (holidays.index <= END_DATE_P1D)
holidays = holidays.loc[mask]
holidays = holidays.resample(rule='H').pad()
holidays = holidays[:-1]  # drop last row
holidays.dst = holidays.apply(fix_dst, axis='columns')
# print(holidays[holidays.dst.notna()])
print('loaded\n')

# loading dataset
houses = []
print('loading dataset...')
for house in HOUSES:
    houses.append(load_dataset(house))
print('loaded\n')

print('merging...')
merged_houses = merge_all(houses)
print(merged_houses.shape)
print('merged\n')

# print('droping nan...')
# merged_houses.dropna(inplace=True)
# print('droped\n')

# table, rows = missing_values_table(merged_houses)
# print(table)
# print(f'{rows / 1000000:.2}M rows ({rows})')

print('getting data inside time window...')
mask = (merged_houses.index >= START_DATE) & (merged_houses.index <= END_DATE)
merged_houses = merged_houses.loc[mask]
# print(merged_houses.count())
# print(merged_houses)
print(merged_houses.shape)
print('got\n')

# print('processing...')
# only_nan = merged_houses[merged_houses.isna().any(axis='columns')]
# print('processed\n')

# print(only_nan.count())
# print(only_nan)

print('summing data...')
merged_houses['kWh_total'] = merged_houses.sum(axis='columns')
merged_houses['kWh_missing'] = merged_houses.isna().sum(axis='columns')
print(merged_houses.shape)
print('summed\n')

print('dropping kWh by house...')
merged_houses.drop(columns=[f'kWh_{house}' for house in HOUSES], inplace=True)
print('dropped')

# TODO add solar and weather
final = merge_all([holidays, weather, merged_houses])
# print(final.shape)
# print(final)
# final.info()
# print(final[final.missing > 0])
# print(final[final.missing > 1].count())
# print(final[final.dst.notna()])
final.to_csv('dataset.csv')
