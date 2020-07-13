import pandas as pd
from numpy import nan, isnan, mean
from datetime import timedelta
import os

PATH = 'dataset'
FINAL = os.path.join(PATH, 'final.csv')
HUE = os.path.join(PATH, 'hue')
WEATHER = os.path.join(HUE, 'Weather_YVR.csv')
HOLIDAYS = os.path.join(HUE, 'Holidays.csv')

START_DATE = '2015-09-29 00:00:00'
END_DATE = '2018-01-29 23:00:00'
END_DATE_P1D = '2018-01-30 23:00:00'  # plus 1 day
HOUSES = (
            # houses located in YVR
            # Vancouver and Lower Mainland area
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20,

            # house located in WYJ
            # Victoria and surrounding area
            # 15,

            # 0 or few data between 2015~2018
            # 1, 17, 26, 27, 28,

            # many missing data between 2015~2018
            # 16, 21, 23, 24, 25,

            # some missing data between 2015~2018
            # 2, 22,

            # almost no missing data between 2015~2018
            # 8, 11, 12, 18,
)


def fix_dst(row):
    if isnan(row.dst) or row.name.hour != 1:
        return 0
    return row.dst


def fix_holiday(value):
    if type(value) == float:
        return 0
    return value


def fix_kWh(column, column_name=None, should_print=True):
    if column_name:
        if column.name != column_name:
            return column
    new_column = column.copy(deep=True)

    if should_print:
        print(f'> filling {new_column.name}...')
    for index, value in new_column.iteritems():
        if isnan(value):
            various = []
            for days in range(1, 30):
                lookup_date = pd.to_timedelta(days, unit='D')
                try:
                    lookup_value = new_column[index + lookup_date]
                except KeyError:
                    lookup_value = nan
                if isnan(lookup_value):
                    try:
                        lookup_value = new_column[index - lookup_date]
                    except KeyError:
                        lookup_value = nan
                if not isnan(lookup_value):
                    various.append(lookup_value)
            value = mean(various, dtype=new_column.dtype)
            new_column[index] = value
    return new_column


def load_dataset(number):
    print(f'House {number:2}', end=': ')
    dataset_path = os.path.join(HUE, f'Residential_{number}.csv')
    dataframe = pd.read_csv(dataset_path, parse_dates=[0])
    print(dataframe.shape, end=' -> ')
    dataframe.rename(columns={'energy_kWh': f'kWh_{number}'}, inplace=True)
    dataframe.hour = dataframe.hour.apply(lambda hour: timedelta(hours=hour))
    dataframe['datetime'] = dataframe.date + dataframe.hour

    mask = (dataframe.datetime >= START_DATE) & \
           (dataframe.datetime <= END_DATE)
    dataframe = dataframe.loc[mask]

    dataframe.set_index(
        'datetime', inplace=True,
        verify_integrity=False  # TODO Keep duplicate, fix this!
    )
    # isso ta bugando, com integrity false
    # dai to removendo o duplicado, mas isso não é o ideal
    trimmed_df = dataframe[[f'kWh_{number}']]
    # ~ == not
    deduplicated = trimmed_df[~trimmed_df.index.duplicated()]
    print(deduplicated.shape)
    return deduplicated


def merge_all(houses):
    left = houses.pop(0)
    while houses:
        right = houses.pop(0)
        left = pd.merge(
            left, right, how='outer',
            left_index=True, right_index=True)
    print(left.shape)
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
weather = pd.read_csv(WEATHER, parse_dates=[0])
print(weather.shape, end=' -> ')
weather.hour = weather.hour.apply(lambda hour: timedelta(hours=hour))
weather['datetime'] = weather.date + weather.hour
weather.set_index('datetime', inplace=True, verify_integrity=True)
weather.drop(columns=['date', 'hour'], inplace=True)
mask = (weather.index >= START_DATE) & (weather.index <= END_DATE)
weather = weather.loc[mask]
weather.weather.fillna(method='bfill', inplace=True)
print(weather.shape)
print('loaded\n')

print('loading holidays...')
# dst -> Daylight Savings Time
holidays = pd.read_csv(HOLIDAYS, parse_dates=[0])
print(holidays.shape, end=' -> ')
holidays.set_index('date', inplace=True, verify_integrity=True)
mask = (holidays.index >= START_DATE) & (holidays.index <= END_DATE_P1D)
holidays = holidays.loc[mask]
holidays = holidays.resample(rule='H').pad()
holidays = holidays[:-1]  # drop last row
holidays.dst = holidays.apply(fix_dst, axis='columns')
holidays.holiday = holidays.holiday.apply(fix_holiday)
holidays.rename(columns={'day': 'day_of_week'}, inplace=True)
print(holidays.shape)
print('loaded\n')

# loading dataset
houses = []
print('loading dataset...')
for house in HOUSES:
    houses.append(load_dataset(house))
print('loaded\n')

print('merging houses...')
print('(0, 0)', end=' -> ')
merged_houses = merge_all(houses)
print('merged\n')

# table, rows = missing_values_table(merged_houses)
# print(table)
# print(f'{rows / 1000000:.2}M rows ({rows})')

# print('processing...')
# only_nan = merged_houses[merged_houses.isna().any(axis='columns')]
# print('processed\n')

# print(only_nan.count())
# print(only_nan)

print('filling NaN...')
before = merged_houses.shape
merged_houses = merged_houses.apply(fix_kWh)
print(f'{before} -> {merged_houses.shape}')
del before
print('filled\n')

print('summing data...')
print(merged_houses.shape, end=' -> ')
merged_houses['kWh'] = merged_houses.sum(axis='columns')
# merged_houses['kWh_missing'] = merged_houses.isna().sum(axis='columns')
print(merged_houses.shape)
print('summed\n')

print('dropping kWh by house...')
print(merged_houses.shape, end=' -> ')
merged_houses.drop(columns=[f'kWh_{house}' for house in HOUSES], inplace=True)
print(merged_houses.shape)
print('dropped\n')

print('merging houses with other data...')
print(merged_houses.shape, end=' -> ')
final = merge_all([holidays, weather, merged_houses])
print('merged\n')

print('filling NaN...')
print(final.shape, end=' -> ')
final.temperature.fillna(method='bfill', inplace=True)
final.humidity.fillna(method='bfill', inplace=True)
final.pressure.fillna(method='bfill', inplace=True)
final.weather.fillna(method='bfill', inplace=True)
final = final.apply(fix_kWh, column_name='kWh', should_print=False)
print(final.shape)
print('filled\n')

print('creating date columns...')
final['year'] = final.index.year
final['month'] = final.index.month
final['day'] = final.index.day
final['hour'] = final.index.hour
print(final)
print('created\n')

final.info()

print('\nsaving to file...')
final.to_csv(FINAL)
print('saved')
