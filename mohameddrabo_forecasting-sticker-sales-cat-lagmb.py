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


!pip install holidays


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import  TransformerMixin
from sklearn.preprocessing import  MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import random
from shapely.wkt import loads
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble  import VotingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import holidays


df =  pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


df['num_sold_log'] = np.log1p(df['num_sold'])


def date_features(data, date_column):
    df  = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    df['year'] = df[date_column].dt.year.astype('int')
    df['quarter'] = df[date_column].dt.quarter.astype('int')
    df['month'] = df[date_column].dt.month.astype('int')
    df['day'] = df[date_column].dt.day.astype('int')
    df['day_of_week'] = df[date_column].dt.dayofweek.astype('int')
    df['week_of_year'] = df[date_column].dt.isocalendar().week.astype('int')
    df['hour'] = df[date_column].dt.hour.astype('int')
    df['minute'] = df[date_column].dt.minute.astype('int')
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7)
    
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    
    return df

def create_holiday(row):
    countries_holiday = holidays.country_holidays(row['country'])
    return 1 if row['date'] in countries_holiday else 0



df =  date_features(df, 'date')
df['holiday'] = df.apply(create_holiday,axis=1)


class Custom_Scaler(TransformerMixin):
    def __init__(self, except_col=[], cols=[], strategy="MinMax"):
        super().__init__()
        self.except_col=except_col
        self.cols = cols if cols else []
        self.strategy = strategy

    def fit(self, df):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        final_col =  numerical_cols.difference(self.except_col)
        self.col  =  final_col if not self.cols else self.cols
        self.scaler = MinMaxScaler().fit(df[self.col]) if self.strategy=="MinMax" else StandardScaler().fit(df[self.col])
        return self
    
    def transform(self, data):
        df =data.copy()
        scaler_data =  self.scaler.transform(df[self.col])
        scaler_data_df = pd.DataFrame(scaler_data, columns=self.col, index=df.index)
        others_cols  =  df.columns.difference(self.col)
        return pd.concat([scaler_data_df, df[others_cols]], axis='columns')

class CustomOneHotEncoder(TransformerMixin):
    def __init__(self, except_col=[], cols=[]):
        super().__init__()
        self.except_col=except_col
        self.cols = cols if cols else []

    def fit(self, data):
        df =  data.copy()
        cat_col =  df.select_dtypes(exclude=[np.number]).columns
        final_col =  cat_col.difference(self.except_col)
        self.col  =  final_col if not self.cols else self.cols
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist'), self.col)
            ],
            remainder='passthrough'  # To keep other columns unchanged
        )
        self.preprocessor =  preprocessor
        self.preprocessor.fit(df[self.col])
        return self

    def transform(self, data):
        df =  data.copy()
        final_data_encoded =  self.preprocessor.transform(df[self.col])
        feature_names = (self.preprocessor
                        .named_transformers_['cat']
                        .get_feature_names_out(self.col))
        final_data_encoded_df = pd.DataFrame(final_data_encoded.toarray() if type(final_data_encoded)!=np.ndarray else final_data_encoded, columns=feature_names, index=df.index)
        others_col =  df.columns.difference(self.col)
        final_df  = pd.concat([df[others_col], final_data_encoded_df], axis='columns')
        return final_df

class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self, except_col=[]):
        self.except_col = except_col
        self.label_encoders = defaultdict(LabelEncoder)

    def fit(self,X , y=None):
        df  = X.copy()
        cat_col =  df.select_dtypes(exclude=[np.number]).columns
        final_col =  cat_col.difference(self.except_col)
        self.columns = final_col
        for col in self.columns:
            self.label_encoders[col]
            self.label_encoders[col].fit(df[col])
        return self

    def transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = X_copy[col].apply(lambda s: '<unknown>' if s not in self.label_encoders[col].classes_ else s)
            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, '<unknown>')
            X_copy[col] = self.label_encoders[col].transform(X_copy[col])
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = self.label_encoders[col].inverse_transform(X_copy[col])
        return X_copy


pipe  = Pipeline([('label_encoding', MultiColumnLabelEncoder(except_col=['date']))])


transform_data  = pipe.fit_transform(df)


transform_data.dropna(inplace=True)


X = transform_data.drop(['num_sold_log', 'id', 'num_sold', 'date'], axis='columns')
y = transform_data['num_sold_log']


LGBM_params  = {'learning_rate': 0.05084033772215662, 'n_estimators': 669, 'max_depth': 5, 'num_leaves': 20, 'min_child_samples': 93, 'subsample': 0.8534926285182945, 'colsample_bytree': 0.6753785281047793, 'reg_alpha': 0.18192200777699893, 'reg_lambda': 8.938165769296162, 'verbose': -1}


test  = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


test = date_features(test, 'date')
test['holiday'] = test.apply(create_holiday, axis=1)


test_transform =  pipe.transform(test)


test_transform.drop(['id', 'date'], axis='columns', inplace=True)


scores = []
prediction = []
model = LGBMRegressor(**LGBM_params)
# Entraînement
model.fit(X, y)
# Prédiction
# y_pred1 = model.predict(X_test)
    
# Calcul du score
# score1 = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(y_pred1))
# print(f"score model 1 : {score1}")
# scores.append(score1)
prediction= model.predict(test_transform)
# Aficher les résultats
# print(f"Scores pour chaque fold model 1 : {scores}")
# print(f"Score moyen model 2: {np.mean(scores):.4f}±{np.std(scores)}")





submission  = pd.DataFrame([], columns=['id', 'num_sold'])
submission.id  =  test.id
submission['num_sold']  =np.expm1(prediction)


submission.to_csv('submission.csv', index=False)

