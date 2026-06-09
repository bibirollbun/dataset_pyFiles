# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv')
test = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv')
sample = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/sample_solution.csv')
train.head()


train.describe()


train.info()


train['class'] = train['class'].map({'Economy': 0, 'Business': 1})
train['stops'] = train['stops'].map({'one': 1, 'zero': 0, 'two_or_more': 2})


# o'rtacha qimmat va o'rtacha arzon aviakompaniyalar
train.groupby(['airline']).agg({'price': 'mean'})
# ['AirAsia', 'Indigo', 'GO_FIRST','SpiceJet','Air_India', 'Vistara']


train['airline'] = train['airline'].map({'AirAsia': 0, 'Indigo': 1, 'GO_FIRST': 2, 'SpiceJet': 3, 'Air_India': 4, 'Vistara': 5})
train.head()


train.select_dtypes(include=(int, float)).corrwith(train['price']).abs()
# train.hist(bins=50, figsize=(12,8))
# sns.countplot(train)


train.isnull().sum()


train.shape


train.select_dtypes(include='object').columns


train.select_dtypes(include='number').columns


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

X = train.drop('price', axis=1)
y = train['price']

scaler = StandardScaler()

nums = ['id', 'airline', 'stops', 'class', 'duration', 'days_left']
cats = ['flight', 'source_city', 'departure_time', 'arrival_time','destination_city']

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')


full_pipeline = ColumnTransformer(
    transformers=[
        ('num', num_transformer, nums),  # Raqamli ustunlarni o'zgartirish
        ('cat', cat_transformer, cats)   # Kategorik ustunlarni o'zgartirish
    ])

# df = pd.get_dummies(df, columns=cats)
X_prepared = full_pipeline.fit_transform(X)
# X_prepared = X_prepared.toarray()
X_train, X_valid, y_train, y_valid = train_test_split(X_prepared, y, test_size=0.2, random_state=42)


X_train.toarray()
X_valid.toarray()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

def score(y_train, y_pred):
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    # EVS hisoblash
    evs = explained_variance_score(y_train, y_pred)
    print(f'MAE: {mae}')
    print(f'RMSE: {np.sqrt(mse)}')
    print(f'R-squared: {r2}')
    print(f"Explained Variance Score: {evs}")


# LinearRegression
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression().fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_valid)

print('LinearRegression (valid set)\n')
score(y_valid, lr_y_pred)


# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_valid)

print('RandomForestRegressor:\n')
score(y_valid, rf_y_pred)


# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
tree_y_pred = tree_model.predict(X_valid)

print('DecisionTreeRegressor:\n')
score(y_valid, tree_y_pred)


import xgboost as xgb

xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Modeldan bashorat qilish
y_pred_xgb = xgb_model.predict(X_valid)

# Natijalarni baholash
# XGB
print('XGBRegressor\n')
score(y_valid, y_pred_xgb)



test['class'] = test['class'].map({'Economy': 0, 'Business': 1})
test['stops'] = test['stops'].map({'one': 1, 'zero': 0, 'two_or_more': 2})

test['airline'] = test['airline'].map({'AirAsia': 0, 'Indigo': 1, 'GO_FIRST': 2, 'SpiceJet': 3, 'Air_India': 4, 'Vistara': 5})


test_prepared = full_pipeline.transform(test)
test_prepared.toarray()


y_pred = rf_model.predict(test_prepared)


sample['price'] = y_pred


sample.head()


sample.to_csv('aviachiptabashorat_new.csv', index=False)


test




