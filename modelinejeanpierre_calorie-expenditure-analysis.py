import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # kde plots

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyRegressor # mean
from sklearn.linear_model import LinearRegression # linear
from sklearn.ensemble import RandomForestRegressor # ensemble 
from sklearn.metrics import mean_squared_log_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# check column name and data type
print("\nColumn Data Type:")
print(train_df.dtypes)


print(train_df.describe(include='all').round(3)) # describe numeric and object data


print(train_df.isna().sum()) # check missing data


# Count unique values in id -> no duplicates
num_unique = train_df['id'].nunique()
print(num_unique)


train_df = train_df.drop('id', axis=1)


# distribution of the features and target

numeric_cols = train_df.select_dtypes(include=['number']).columns

for column in numeric_cols:
    sns.displot(train_df, x=column, kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


sexEncoded = pd.get_dummies(train_df['Sex'], dtype = float)
sexEncoded.rename(columns={'female':'Sex_Female'}, inplace=True)


trainEncoded_df = pd.concat([train_df, sexEncoded], axis=1) # add numeric gender to df


trainEncoded_df.drop(['Sex', 'male'], axis=1, inplace=True) # drop to have 1 col for Sex


trainEncoded_df.rename(columns={'Sex_Female':'Sex'},inplace=True)


print('\nPearson Correlation:')
trainEncoded_df.corr('pearson')


print(trainEncoded_df['Sex'].value_counts())


# log transform the target to reduce skewness
trainEncoded_df['Calories_log'] = np.log1p(trainEncoded_df['Calories'])


print(trainEncoded_df.head(5))


# drop calories col since we now have its log
trainEncoded_df = trainEncoded_df.drop(columns='Calories', axis=1)


# define features as x and target as y
X=trainEncoded_df.drop(columns=['Calories_log'])
y=trainEncoded_df['Calories_log']


# splitting training and val set 80/20
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Dummy Regressor Model (Baseline)

dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit(X_train, y_train)

# predict on val
dummy_vals = dummy_model.predict(X_val)

# RMSLE performance
dummy_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(dummy_vals)))
print("Dummy Regressor RMSLE:",dummy_rmsle)


# Linear Regression Model

LR_model = LinearRegression()
LR_model.fit(X_train, y_train)

# predict on val
LR_vals = LR_model.predict(X_val)

# RMSLE performance
LR_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(LR_vals)))
print("Linear Regression RMSLE:",LR_rmsle)


# Random Forest Model

RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)

# predict on val
RF_vals = RF_model.predict(X_val)

# RMSLE performance
RF_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(RF_vals)))
print("Random Forest RMSLE:",RF_rmsle)


# check missing and duplicates
print(test_df.shape)
print('\nNull:\n',test_df.isnull().sum())
duplicates_count = test_df.duplicated().sum()
print('\nDuplicates:',duplicates_count)


# remove id column 
test_df = test_df.drop('id', axis=1)


# encode sex, 0=male, 1=female
sexEncoded = pd.get_dummies(test_df['Sex'], dtype = float)
sexEncoded.rename(columns={'female':'Sex_Female'}, inplace=True)


# concat to test df
testEncoded_df = pd.concat([test_df, sexEncoded], axis=1)


# drop male, female col
testEncoded_df.drop(['Sex', 'male'], axis=1, inplace=True)
# rename sex_female
testEncoded_df.rename(columns={'Sex_Female':'Sex'},inplace=True)
print(testEncoded_df.head(10))


# X val for predictions
X_test = testEncoded_df


# make predictions using random forest regressor model

test_preds = RF_model.predict(X_test)
test_preds = np.expm1(test_preds) # reverse log


sample_df['Calories'] = test_preds.round(1)
sample_df.to_csv("my_submission.csv", index=False)


# visualize test predicts

sns.displot(pd.Series(test_preds), bins=30, color='orange', kde=True)

plt.xlabel('Predicted Calories Burned')
plt.ylabel('Frequency')
plt.title('Predicted Calorie Distribution')
plt.show()

