
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading data
data = pd.read_csv('bike-train.csv')

data.head()

nrows = data.shape[0]
ncols = data.shape[1]

print('In this sheet, we have {0} rows and {1} columns'.format(nrows, ncols))

data.info()

data['datetime'] = pd.to_datetime(data['datetime'])

data.info()

plt.figure()
data[:24 * 10].plot(x='datetime', y='count')

# Preparing the data
season_dummies = pd.get_dummies(data['season'], prefix='season', drop_first=False)
data = pd.concat([data, season_dummies], axis=1)

weather_dummies = pd.get_dummies(data['weather'], prefix='weather', drop_first=False)
data = pd.concat([data, weather_dummies], axis=1)

cols_to_drop = ['weather', 'season', 'casual', 'registered']
data.drop(cols_to_drop, inplace=True, axis=1)

quant_features = ['temp', 'humidity', 'windspeed']

scaled_features = {}

for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std

for quant in quant_features:
    mean, std = data[quant].mean(), data[quant].std()
    print('Mean of {0} = {1} and standard deviation = {2}'.format(quant, mean, std))

data.head()

data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day
data['month'] = data['datetime'].dt.month

data.drop('datetime', inplace=True, axis=1)

data.head()

count_per_hour = data.groupby('hour')['count'].mean()

count_per_hour.head()

plt.figure()
count_per_hour.plot(kind='bar')

count_per_month = data.groupby('month')['count'].mean()
count_per_month.plot(kind="bar")

plt.figure()
count_per_month.plot(kind='bar')

# Structeing the data

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

features = ['holiday', 'workingday', 'temp', 'atemp', 'humidity', 'windspeed', 'season_1', 'season_2', 'season_3',
            'season_4', 'weather_1', 'weather_2', 'weather_3', 'weather_4', 'hour', 'day', 'month']
target = ['count']

scores = cross_val_score(
    X=data[features],
    y=data[target].values,
    estimator=DecisionTreeRegressor(),
    scoring='neg_mean_squared_error',
    cv=5
)

scores.mean()
model = DecisionTreeRegressor()
model.fit(data[features], data[target])

# Submitting

test = pd.read_csv('bike-test.csv')

test['datetime'] = pd.to_datetime(test['datetime'])

season_dummies = pd.get_dummies(test['season'], prefix='season', drop_first=False)
test = pd.concat([test, season_dummies], axis=1)

weather_dummies = pd.get_dummies(test['weather'], prefix='weather', drop_first=False)
test = pd.concat([test, weather_dummies], axis=1)

for each in quant_features:
    data.loc[:, each] = (data[each] - scaled_features[each][0]) / scaled_features[each][1]

test['hour'] = test['datetime'].dt.hour
test['day'] = test['datetime'].dt.day
test['month'] = test['datetime'].dt.month

predictions = model.predict(test[features])
test['count'] = predictions
test[['datetime', 'count']].head()

test[['datetime', 'count']].to_csv('submission.csv', index=False)
