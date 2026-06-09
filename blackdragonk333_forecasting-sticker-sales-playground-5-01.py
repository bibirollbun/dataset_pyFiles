import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


train_df.shape


train_df.sample(5)


train_df.columns


train_df.info()


train_df.isna().sum()


train_df.duplicated().sum()


train_df.corr(numeric_only=True)


train_df['date'] = pd.to_datetime(train_df['date'])


def show_count(col):
    sns.countplot(train_df, x=train_df[col])
    plt.title(f'Count plot of {col}')
    plt.show()


for col in train_df.select_dtypes(include=['object']):
    show_count(col)


sns.countplot(train_df, x='product', hue='store')


sns.countplot(train_df, x='product', hue='country')


train_df.head()


train_df_no_na = train_df.copy()


train_df_no_na.info()


train_df_no_na.dropna(inplace=True)


train_df_no_na.info()


train_df_no_na['year'] = train_df_no_na['date'].dt.year
train_df_no_na['month'] = train_df_no_na['date'].dt.month
train_df_no_na['day'] = train_df_no_na['date'].dt.day


train_df_no_na.drop(columns=['date'], inplace=True)


train_df_no_na.drop(columns=['id'], inplace=True)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor


rf_reg = RandomForestRegressor()


X = train_df_no_na.drop(columns=['num_sold'])


y = train_df_no_na['num_sold']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


encoder = Pipeline(
    [
        ('one_hot_enc', OneHotEncoder(drop='first'))
    ]
)


X_train.head()


preprocessor = ColumnTransformer(
    [
        ('encode', encoder, [0, 1, 2])
    ]
)


rf_pipe = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('model', RandomForestRegressor())
    ]
)


lin_reg_pipe = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('model', LinearRegression())
    ]
)


SGDR_pipe = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('model', SGDRegressor())
    ]
)


GB_pipe = Pipeline(
    [
        ('preprocessing', preprocessor),
        ('model', GradientBoostingRegressor())
    ]
)


rf_pipe.fit(X_train, y_train)


lin_reg_pipe.fit(X_train, y_train)


SGDR_pipe.fit(X_train, y_train)


GB_pipe.fit(X_train, y_train)


y_pred_rf = rf_pipe.predict(X_test)


y_pred_lin_reg = lin_reg_pipe.predict(X_test)


y_pred_SGDR = SGDR_pipe.predict(X_test)


y_pred_GB = GB_pipe.predict(X_test)


mean_absolute_percentage_error(y_test, y_pred_rf)


print('Linear Regression:', mean_absolute_percentage_error(y_test, y_pred_lin_reg))


print('SGD REgressor:', mean_absolute_percentage_error(y_test, y_pred_SGDR))


print('Gradient Boosting:', mean_absolute_percentage_error(y_test, y_pred_GB))


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


ids = test['id']
ids


test.info()


test['date'] = pd.to_datetime(test['date']) 


test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day


test.drop(columns=['id', 'date'], inplace=True)


test_pred_rf = rf_pipe.predict(test)


test_pred_lin_reg = lin_reg_pipe.predict(test)


test_pred_GB = GB_pipe.predict(test)


result_rf = pd.DataFrame({'id': ids, 'num_sold': test_pred_rf})
result_rf


result_lin_reg = pd.DataFrame({'id': ids, 'num_sold': test_pred_lin_reg})
result_lin_reg


result_GB = pd.DataFrame({'id': ids, 'num_sold': test_pred_GB})
result_GB


result_rf.to_csv('Random_Forest_by_removing_missing_values.csv', index=False)


result_lin_reg.to_csv('Linear_Regression_by_removing_missing_values.csv', index=False)


result_GB.to_csv('Gradient_Boosting_by_removing_missing_values.csv', index=False)




