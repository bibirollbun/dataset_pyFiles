
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train_df.info()


train_df.sample(5)


train_df.drop(columns=['id'], inplace=True)


train_df.isna().sum()


train_df.dropna(inplace=True)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


train_df.head()


num = Pipeline(
    [
        ('impute', SimpleImputer(strategy='mean'))
    ]
)

cat = Pipeline(
    [
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(drop='first'))
    ]
)


preprocessor = ColumnTransformer(
    [
        ('num', num, [3, 8]),
        ('cat', cat, [0, 1, 2, 4, 5, 6, 7])
    ],
    remainder='passthrough'
)


X = train_df.drop(columns='Price')
y = train_df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


lin_reg = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ]
)


lin_reg.fit(X_train, y_train)


y_pred_lin_reg = lin_reg.predict(X_test)


np.sqrt(mean_squared_error(y_test, y_pred_lin_reg))


rf = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor())
    ]
)

GB = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor())
    ]
)


rf.fit(X_train, y_train)


y_pred_rf = rf.predict(X_test)


print('rf:', mean_squared_error(y_test, y_pred_rf) ** 0.5)


GB.fit(X_train, y_train)


y_pred_GB = GB.predict(X_test)
print('GB:', mean_squared_error(y_test, y_pred_GB) ** 0.5)


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


test.head()


ids = test['id']
test.drop(columns='id', inplace=True)


mean_impute = SimpleImputer(strategy='mean')
freq_impute = SimpleImputer(strategy='most_frequent')


test_pred_GB = GB.predict(test)


result_GB = pd.DataFrame({'id': ids, 'Price': test_pred_GB})
result_GB


result_GB.to_csv('submission.csv', index=False)




