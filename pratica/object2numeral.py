import pandas as pd

DAYS = {
    'Sunday': 1,
    'Monday': 2,
    'Tuesday': 3,
    'Wednesday': 4,
    'Thursday': 5,
    'Friday': 6,
    'Saturday': 7,
}

WEATHERS = {
   'Clear': 1,
   'Mainly Clear': 2,
   'Fog': 3,
   'Cloudy': 4,
   'Mostly Cloudy': 5,
   'Rain': 6,
   'Moderate Rain Showers': 7,
   'Rain Showers': 8,
   'Moderate Rain': 9,
   'Drizzle': 10,
   'Freezing Fog': 11,
   'Snow Showers': 12,
   'Snow': 13,
   'Heavy Rain': 14,
   'Heavy Rain Showers': 15,
   'Thunderstorms': 16,
   'Moderate Snow': 17,
   'Ice Pellets': 18,
   'Freezing Rain': 19
}

HOLIDAYS = {
    '0': 0,
    'Thanksgiving': 1,
    'Remembrance Day': 2,
    'Christmas Day': 3,
    'Boxing Day': 4,
    'New Years': 5,
    'Family Day': 6,
    'Good Friday': 7,
    'Easter Monday': 8,
    'Victoria Day': 9,
    'Canada Day': 10,
    'Civic Day': 11,
    'Labour Day': 12,
}

dataset = pd.read_csv('dataset.csv', index_col=0, parse_dates=[0])

dataset.day_of_week = dataset.day_of_week.apply(lambda day: DAYS[day])
dataset.holiday = dataset.holiday.apply(lambda holiday: HOLIDAYS[holiday])
dataset.weather = dataset.weather.apply(lambda weather: WEATHERS[weather])
dataset.dst = dataset.dst.apply(int)

dataset.to_csv('dataset_numeral.csv')

print(f'Day labels\n{DAYS}\n')
print(f'Holiday labels\n{HOLIDAYS}\n')
print(f'Weather labels\n{WEATHERS}')
