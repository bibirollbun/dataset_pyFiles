
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

import os
    
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 'id')
train.head()
test.head()
df = pd.concat([train, test])
# df = clean(df)
# df = encode(df)
for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    
train = df.loc[train.index, :]
test = df.loc[test.index, :]

train.dropna(axis=0, subset=['rainfall','day'], inplace=True)
y = train.rainfall
train.drop(['rainfall', 'day'], axis=1, inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(train, y, 
                                                                train_size=0.7, test_size=0.3,
                                                                random_state=0)
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]
print(numerical_cols)
# my_cols = categorical_cols + numerical_cols
my_cols = numerical_cols
# my_cols = ['humidity', 'cloud', 'sunshine']
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = test[my_cols].copy()





from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))])

my_pipeline.fit(X_train, y_train)
preds_test = my_pipeline.predict(X_test)

output = pd.DataFrame({'id': X_test.index,
                       'rainfall': preds_test})
output.to_csv('submission.csv', index=False)


numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer =  Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols)
        # ('cat', categorical_transformer, categorical_cols)
    ])
# model = RandomForestRegressor(n_estimators=100, random_state=0)
# model = XGBRegressor()
model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', model)])

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)

score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

