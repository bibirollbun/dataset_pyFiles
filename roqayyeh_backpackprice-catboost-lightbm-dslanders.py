# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Basic Libraries

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
orig = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')


train = train.drop('id', axis=1)


print(f'train shape: {train.shape}')


train.info()


train.head()


train.describe().T


train.describe(include='O')


for col in train.select_dtypes('object').columns:
    print(f'Column Name: {col}')
    print(f'Number of Unique Values: {train[col].nunique()}')
    print(train[col].value_counts())
    print('*********************')


train.isnull().sum()


train.isnull().sum().sum()


train[train.isnull().any(axis=1)]


train.duplicated().sum()


train['Compartments'] = train['Compartments'].astype(int)


train.columns


print('Compartment column skewness:', train['Compartments'].skew())
print('Weight Capacity column skewness:',train['Weight Capacity (kg)'].skew())
print('Price column skewness:', train['Price'].skew())


test_id = test['id']


test = test.drop('id', axis=1)


print(f'test shape: {test.shape}')


test.info()


test.head()


test.describe()


test.describe(include='O')


for col in test.select_dtypes('object').columns:
    print(f'Column Name: {col}')
    print(f'Number of Unique Values: {test[col].nunique()}')
    print(test[col].value_counts())
    print('*********************')


test.isnull().sum()


test.isnull().sum().sum()


test[test.isnull().any(axis=1)]


test.duplicated().sum()


test['Compartments'] = test['Compartments'].astype(int)


print('Compartment column skewness:', test['Compartments'].skew())
print('Weight Capacity column skewness:',test['Weight Capacity (kg)'].skew())


print(f'original set shape: {orig.shape}')


orig.info()


orig.head()


orig.describe()


orig.describe(include='O')


for col in orig.select_dtypes('object').columns:
    print(f'Column Name: {col}')
    print(f'Number of Unique Values: {orig[col].nunique()}')
    print(orig[col].value_counts())
    print('*********************')


orig.isnull().sum()


orig.isnull().sum().sum()


orig[orig.isnull().any(axis=1)]


print('Compartment column skewness:', orig['Compartments'].skew())
print('Weight Capacity column skewness:',orig['Weight Capacity (kg)'].skew())
print('Price column skewness:', orig['Price'].skew())


orig['Compartments'] = orig['Compartments'].fillna(orig['Compartments'].mean()).astype(int)


cat_cols = [col for col in test.select_dtypes('object')]
num_cols = [col for col in train.select_dtypes('number')]


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(10,5))
index = 0
axs = axs.flatten()

for col in num_cols:
    sns.histplot(train[col], bins=50, ax=axs[index])
    index += 1

    
plt.tight_layout();


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(10,6))
index = 0
axs = axs.flatten()

for col in num_cols:
    sns.boxplot(y=col,data=train, ax=axs[index])
    index += 1
    
plt.tight_layout();


fig, axs = plt.subplots(nrows=4,ncols=2,figsize=(12,10))
axs = axs.flatten()


for index, col in enumerate(cat_cols):
    sns.countplot(data=train, x=col, ax=axs[index])

axs[len(cat_cols)].remove()

plt.tight_layout()
plt.show();


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
index = 0
axs = axs.flatten()

for col in test.select_dtypes('number').columns:
    sns.histplot(test[col], bins=50, ax=axs[index])
    index += 1

    
plt.tight_layout();


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(10,6))
index = 0
axs = axs.flatten()

for col in test.select_dtypes('number').columns:
    sns.boxplot(y=col,data=train, ax=axs[index])
    index += 1
    
plt.tight_layout();


fig, axs = plt.subplots(nrows=4,ncols=2,figsize=(12,10))
axs = axs.flatten()


for index, col in enumerate(cat_cols):
    sns.countplot(data=test, x=col, ax=axs[index])

axs[len(cat_cols)].remove()

plt.tight_layout()
plt.show();


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(10,5))
index = 0
axs = axs.flatten()

for col in num_cols:
    sns.histplot(orig[col], bins=50, ax=axs[index])
    index += 1

    
plt.tight_layout();


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(10,6))
index = 0
axs = axs.flatten()

for col in num_cols:
    sns.boxplot(y=col,data=orig, ax=axs[index])
    index += 1
    
plt.tight_layout();


fig, axs = plt.subplots(nrows=4,ncols=2,figsize=(12,10))
axs = axs.flatten()


for index, col in enumerate(cat_cols):
    sns.countplot(data=orig, x=col, ax=axs[index])

axs[len(cat_cols)].remove()

plt.tight_layout()
plt.show();


#merging train + orinal set

set(train.columns)-set(orig.columns), set(orig.columns)-set(train.columns)


data = pd.concat([train, orig], axis=0)
data.shape


data.describe(include='O')


data.isna().sum()


data.columns


#Simple null handling for categorical features - fill with mode

for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)
    


#Simple null handling for numerical features - fill with mean_since the columns aren't skewed

data['Price'].fillna(data['Price'].mean(), inplace=True)
data['Weight Capacity (kg)'].fillna(data['Weight Capacity (kg)'].mean(), inplace=True)

test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean(), inplace=True)




data.rename(columns={'Weight Capacity (kg)': 'weight_capacity'}, inplace=True)
test.rename(columns={'Weight Capacity (kg)': 'weight_capacity'}, inplace=True)



data['Compartments_per_Weight'] = data['Compartments'] / data['weight_capacity']
test['Compartments_per_Weight'] = test['Compartments'] / test['weight_capacity']


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error,r2_score,mean_squared_log_error


X = data.drop('Price', axis=1)
y= data['Price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test.copy()


cat_features = data.select_dtypes(include=['object']).columns.tolist()

# Initialize CatBoost model
catboost_model = CatBoostRegressor(iterations=500,
                                   depth=6,
                                   learning_rate=0.05,
                                   loss_function='RMSE',
                                   cat_features=cat_features,
                                   verbose=100)

# Fit model
catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, verbose=100)


val_preds = catboost_model.predict(X_val)


rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print(f'RMSE: {rmse:.4f}')


test_preds = catboost_model.predict(X_test)


output = pd.DataFrame({'id': test_id, 'Price':test_preds})
output.to_csv('submission.csv', index=False)


output.head()


data_1 = data.copy()
test_1 = test.copy()


for col in data_1.select_dtypes(include=['object']).columns:
    data_1[col] = data_1[col].astype('category')
    test_1[col] = test_1[col].astype('category')

X = data_1.drop('Price', axis=1)
y = data_1['Price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_1.copy()


# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=data_1.select_dtypes(include=['category']).columns.tolist())

# Train LightGBM model
lgbm_model = lgb.train({'objective': 'regression', 'metric': 'rmse'}, train_data, num_boost_round=500)


y_pred = lgbm_model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f'RMSE: {rmse:.4f}')


test_preds_lgb = lgbm_model.predict(X_test)


output = pd.DataFrame({'id': test_id, 'Price':test_preds_lgb})
output.to_csv('submission_lgb_1.csv', index=False)


#merged result

merge_preds = (test_preds_lgb + test_preds) / 2

output = pd.DataFrame({'id': test_id, 'Price':merge_preds})
output.to_csv('submission_merged.csv', index=False)

