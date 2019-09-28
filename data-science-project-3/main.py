


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#Reading Data

train_data = pd.read_csv('train.csv')
train_ids = train_data['Id']

test_data = pd.read_csv('test.csv')
test_ids = test_data['Id']

y = train_data['SalePrice']
train_data.drop(columns='SalePrice', inplace=True)

train_data.set_index('Id', inplace=True)
test_data.set_index('Id', inplace=True)

data = pd.concat([train_data, test_data], axis=0)
data.shape

data.head()

#Preparing the data

data.info()
data.isnull().sum()

missing_data = data.isnull().sum()

for k, v in missing_data.items():
    if v > 0:
        print(k, ':\t', v)


categorical_cols = []
numerical_cols = []


for col in data.columns:
    if(data[col].dtypes == 'object'):
        categorical_cols.append(col)
    elif (data[col].dtypes == 'int64' or data[col].dtypes == 'float64'):
        numerical_cols.append(col)



assert len(categorical_cols) == 43, 'Wrong number of categorical columns'
assert len(numerical_cols) == 36, 'Wrong number of neumerical columns'
print('OK! You may proceed.')    




def convert_categorical_to_dummies(data):

    for col in categorical_cols:
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=False, dummy_na=True)
        data = pd.concat([data, dummies], axis=1)
        data.drop(columns=col, inplace=True)

    return data



data = convert_categorical_to_dummies(data)
assert data.shape[1] == 331, 'Wrong shape.'
print('OK! you may proceed.')


def extract_numerical_cols_with_nans(data):
    numerical_cols_w_nans = data[numerical_cols].columns[data[numerical_cols].isnull().sum() > 0].tolist()
    return numerical_cols_w_nans

numerical_cols_w_nans = extract_numerical_cols_with_nans(data)
assert len(numerical_cols_w_nans)==11, 'Numerical Columns with NaNs are in the wrong shape!'


def graph_histogram(cols):
    for index ,col in enumerate(cols, start=1):
        plt.figure(index)
        plt.title(col)
        data[col].plot.hist(bins=40)
        plt.show()



graph_histogram(numerical_cols_w_nans)

def fillna(cols, data):
    for col in cols:
        if col == 'LotFrontage' or col == 'TotalBsmtSF':
            data[col].fillna(data[col].median(), inplace=True)
        else:
            data[col].fillna(-1, inplace=True)
    
    return data


data = fillna(numerical_cols_w_nans, data)

assert (data==-1).sum().sum() == 191, 'Did you fill in missing values the right way ?'
assert data.isnull().sum().sum() == 0, 'There are still NaNs! Task is not completed.'


#Modling Data

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

train_data = data.loc[train_ids]
test_data = data.loc[test_ids]

rf_model = RandomForestRegressor(n_estimators=50)
scores_RF = cross_val_score(rf_model, train_data, y, cv=5, scoring='r2')

scores_RF.mean()

#Submitting

rf_model.fit(train_data, y)

test_data['SalePrice'] = rf_model.predict(test_data)
test_data['SalePrice'].to_csv('house_predictions_submission.csv', header=True)