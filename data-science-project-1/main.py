


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




data = pd.read_csv('titanic.csv')

data.head()
data.info()

[nrows, ncols] = data.shape


c_dataType = data.dtypes
cols_with_missing_vals = data.columns[data.isnull().any()]

n_num_cols = 0
n_str_cols = 0


for element in c_dataType:
    if (element == 'int64') or (element == 'float64'):
        n_num_cols = n_num_cols + 1
    else:
        n_str_cols = n_str_cols + 1
        

print('There are {0} rows and {1} columns in this dataset.'.format(nrows, ncols))
print('Of those columns, there are {0} numerical columns and {1} categorical columns.'.format(n_num_cols, n_str_cols))
print('The following columns have missing values: {0}'.format(cols_with_missing_vals))


data.drop(columns='PassengerId', inplace=True)
data.drop(columns='Name', inplace=True)
data.drop(columns='Ticket', inplace=True)

final_column_set = data.columns.tolist()

age_vals = data['Age'].values
age_vals = age_vals[~np.isnan(age_vals)]

sex_vals = data['Sex'].values
survived_vals = data['Survived'].values

mean_age = np.mean(age_vals)



data['Age'].fillna( mean_age, inplace=True)

data.head()

data['Survived'].value_counts()



[dead, survived] = data['Survived'].value_counts()


survived_percentage = survived / (survived + dead) * 100.0
print('About {0}% of people in this dataset have survived'.format(np.round(survived_percentage, 2)))


data['Pclass'].value_counts()

data['Sex'].value_counts()

data['Parch'].value_counts()

data['Embarked'].value_counts()


female_survived = sex_vals[(sex_vals == 'female') & (survived_vals == 1)].size

female_survived_percentage = female_survived / (survived + dead) * 100.0
print('nrows={}\nncols={}\nn_num_cols={}\nn_str_cols={}\nfinal_column_set={}\ncols_with_missing_vals={}\nmean_age={}\nsurvived_percentage={}\nfemale_survived_percentage={}'.format(
nrows,
ncols,
n_num_cols,
n_str_cols,
final_column_set,
cols_with_missing_vals,
mean_age,
survived_percentage,
female_survived_percentage
))