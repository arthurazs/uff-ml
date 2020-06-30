import pandas as pd
from numpy import nan
from datetime import timedelta

START_DATE = '2015-09-29 00:00:00'
END_DATE = '2018-01-29 23:00:00'
END_DATE_P1D = '2018-01-30 23:00:00'  # plus 1 day


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
    # ~ == ! == not
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
    return left


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mvt_ren_columns = mis_val_table.rename(columns={0: '#', 1: '%'})
    mvt_ren_columns = mvt_ren_columns[
        mvt_ren_columns.iloc[:, 1] >= 0].sort_values(
            '%', ascending=False).round(1)
    return mvt_ren_columns, df.shape[0]


print('loading holidays...')
# dst = Daylight Savings Time
holidays = pd.read_csv('hue/Holidays.csv', parse_dates=[0])
holidays.set_index('date', inplace=True, verify_integrity=True)
mask = (holidays.index >= START_DATE) & (holidays.index <= END_DATE_P1D)
holidays = holidays.loc[mask]
holidays = holidays.resample(rule='H').pad()
holidays = holidays[:-1]  # drop last row
holidays.dst = holidays.apply(fix_dst, axis=1)
# print(holidays[holidays.dst.notna()])
print('loaded\n')

# loading dataset
houses = []
print('loading dataset...')
for house in range(1, 29):
    if house not in (
            # 0 or few data between 2015~2018
            1, 17, 26, 27, 28,
            # many missing data between 2015~2018
            10, 16, 21, 23, 24, 25,
            # some missing data between 2015~2018
            2, 22,
            # almost no missing data between 2015~2018
            # 8, 11, 12, 18
            ):
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
# only_nan = merged_houses[merged_houses.isna().any(axis=1)]
# print('processed\n')

# print(only_nan.count())
# print(only_nan)

print('summing data...')
merged_houses['kWh_total'] = merged_houses.sum(axis=1)
merged_houses['missing'] = merged_houses.isna().sum(axis=1)
print(merged_houses.shape)
print('summed\n')

# TODO add solar and weather
final = merge_all([holidays, merged_houses])
print(final.shape)
# print(final[final.missing > 0])
# print(final[final.dst.notna()])
