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


train_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

sub = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')


from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib import pyplot as plt



%matplotlib inline


lst_cols = list(train_df.columns)
lst_cols


lst_test_cols = test_df.columns.tolist()
lst_test_cols


df_homeless_rate = train_df[['HOMELESS_RATE']]

target = df_homeless_rate


df_ages = train_df[['AGE_U18_PCT',
 'AGE_18_24_PCT',
 'AGE_25_34_PCT',
 'AGE_35_44_PCT',
 'AGE_45_54_PCT',
 'AGE_55_59_PCT',
 'AGE_60_61_PCT',
 'AGE_62_64_PCT',
 'AGE_65_69_PCT',
 'AGE_70_79_PCT',
 'AGE_80_PLUS_PCT',
 'AGE_25_PLUS_PCT',
 'FAMILY_MEMBERS_UNDER_18_PCT',]]

df_ages



df_ages_homeless = pd.concat([df_homeless_rate, df_ages], axis=1)
df_ages_homeless


import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(df_ages_homeless.corr(), annot=True, fmt=".2f", cmap="coolwarm")


df_race = train_df[['HOMELESS_RATE',  'RACE_WHITE_NH_PCT',
 'RACE_BLACK_NH_PCT',
 'RACE_NATIVE_NH_PCT',
 'RACE_ASIAN_NH_PCT',
 'RACE_PACIFIC_NH_PCT',
 'RACE_TWO_OR_MORE_NH_PCT',
 'RACE_HISPANIC_ANY_PCT',]]
df_race


sns.heatmap(df_race.corr(), annot=True, fmt=".2f", cmap="coolwarm")


df_ability = train_df[['HOMELESS_RATE',  'VETERAN_POP_PCT',
 'NONVETERAN_POP_PCT',
 'DISABILITY_POP_PCT',
 'NODISABILITY_POP_PCT',]]

df_ability


sns.heatmap(df_ability.corr(), annot=True, fmt=".2f", cmap="coolwarm")


df_house = train_df[['HOMELESS_RATE', 'TOTAL_HOUSEHOLDS_PCT',]]
df_house


sns.heatmap(df_house.corr(), annot=True, fmt=".2f", cmap="coolwarm")


df_family = train_df[['HOMELESS_RATE',  'FAMILY_HH_TOTAL',
 'FAMILY_HH_CHILD_LT18_PCT',
 'NONFAMILY_SINGLE_MALE_PCT',
 'NONFAMILY_SINGLE_FEMALE_PCT',
 'MULTI_PERSON_NONFAMILY_HH_PCT',
 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT']]

df_family


sns.heatmap(df_family.corr(), annot=True, fmt=".2f", cmap="coolwarm")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def RMSE_LR(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_svr(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = SVR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_RF(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_CatBoost(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = CatBoostRegressor(silent=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_LGBM(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = LGBMRegressor(verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))

def RMSE_XGB(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('HOMELESS_RATE', axis=1), df['HOMELESS_RATE'], test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))




Ages = []

Ages.append(RMSE_LR(df_ages_homeless))
Ages.append(RMSE_svr(df_ages_homeless))
Ages.append(RMSE_RF(df_ages_homeless))
Ages.append(RMSE_CatBoost(df_ages_homeless))
Ages.append(RMSE_LGBM(df_ages_homeless))
Ages.append(RMSE_XGB(df_ages_homeless))

print(RMSE_LR(df_ages_homeless))
print(RMSE_svr(df_ages_homeless))
print(RMSE_RF(df_ages_homeless))
print(RMSE_CatBoost(df_ages_homeless))
print(RMSE_LGBM(df_ages_homeless))
print(RMSE_XGB(df_ages_homeless))
print(np.mean([RMSE_LR(df_ages_homeless), RMSE_svr(df_ages_homeless), RMSE_RF(df_ages_homeless), RMSE_CatBoost(df_ages_homeless), RMSE_LGBM(df_ages_homeless), RMSE_XGB(df_ages_homeless)]))
print('AGES')
print(Ages)



Races = []

Races.append(RMSE_LR(df_race))
Races.append(RMSE_svr(df_race))
Races.append(RMSE_RF(df_race))
Races.append(RMSE_CatBoost(df_race))
Races.append(RMSE_LGBM(df_race))
Races.append(RMSE_XGB(df_race))

print(RMSE_LR(df_race))
print(RMSE_svr(df_race))
print(RMSE_RF(df_race))
print(RMSE_CatBoost(df_race))
print(RMSE_LGBM(df_race))
print(RMSE_XGB(df_race))
print(np.mean([RMSE_LR(df_race), RMSE_svr(df_race), RMSE_RF(df_race), RMSE_CatBoost(df_race), RMSE_LGBM(df_race), RMSE_XGB(df_race)]))
print('RACES')
print(Races)



Ability = []

Ability.append(RMSE_LR(df_ability))
Ability.append(RMSE_svr(df_ability))
Ability.append(RMSE_RF(df_ability))
Ability.append(RMSE_CatBoost(df_ability))
Ability.append(RMSE_LGBM(df_ability))
Ability.append(RMSE_XGB(df_ability))

print(RMSE_LR(df_ability))
print(RMSE_svr(df_ability))
print(RMSE_RF(df_ability))
print(RMSE_CatBoost(df_ability))
print(RMSE_LGBM(df_ability))
print(RMSE_XGB(df_ability))
print(np.mean([RMSE_LR(df_ability), RMSE_svr(df_ability), RMSE_RF(df_ability), RMSE_CatBoost(df_ability), RMSE_LGBM(df_ability), RMSE_XGB(df_ability)]))
print('ABILITY')
print(Ability)


Family = []

Family.append(RMSE_LR(df_family))
Family.append(RMSE_svr(df_family))
Family.append(RMSE_RF(df_family))
Family.append(RMSE_CatBoost(df_family))
Family.append(RMSE_LGBM(df_family))
Family.append(RMSE_XGB(df_family))

print(RMSE_LR(df_family))
print(RMSE_svr(df_family))
print(RMSE_RF(df_family))
print(RMSE_CatBoost(df_family))
print(RMSE_LGBM(df_family))
print(RMSE_XGB(df_family))
print(np.mean([RMSE_LR(df_family), RMSE_svr(df_family), RMSE_RF(df_family), RMSE_CatBoost(df_family), RMSE_LGBM(df_family), RMSE_XGB(df_family)]))
print('FAMILY')
print(Family)



House = []

House.append(RMSE_LR(df_house))
House.append(RMSE_svr(df_house))
House.append(RMSE_RF(df_house))
House.append(RMSE_CatBoost(df_house))
House.append(RMSE_LGBM(df_house))
House.append(RMSE_XGB(df_house))

print(RMSE_LR(df_house))
print(RMSE_svr(df_house))
print(RMSE_RF(df_house))
print(RMSE_CatBoost(df_house))
print(RMSE_LGBM(df_house))
print(RMSE_XGB(df_house))
print(np.mean([RMSE_LR(df_house), RMSE_svr(df_house), RMSE_RF(df_house), RMSE_CatBoost(df_house), RMSE_LGBM(df_house), RMSE_XGB(df_house)]))
print('HOUSE')
print(House)


train_df.drop('ID', axis=1, inplace=True)


Train_df = []

Train_df.append(RMSE_LR(train_df))
Train_df.append(RMSE_svr(train_df))
Train_df.append(RMSE_RF(train_df))
Train_df.append(RMSE_CatBoost(train_df))
Train_df.append(RMSE_LGBM(train_df))
Train_df.append(RMSE_XGB(train_df))

print(RMSE_LR(train_df))
print(RMSE_svr(train_df))
print(RMSE_RF(train_df))
print(RMSE_CatBoost(train_df))
print(RMSE_LGBM(train_df))
print(RMSE_XGB(train_df))
print(np.mean([RMSE_LR(train_df), RMSE_svr(train_df), RMSE_RF(train_df), RMSE_CatBoost(train_df), RMSE_LGBM(train_df), RMSE_XGB(train_df)]))
print('TRAIN')
print(Train_df)


feature_corr =train_df.corr()[train_df.corr()['HOMELESS_RATE'].abs() > 0.2].sort_values(by='HOMELESS_RATE', ascending=False).index


feature_corr


features = [ 'AGE_25_34_PCT', 'NONVETERAN_POP_PCT', 'AGE_35_44_PCT',
       'RACE_BLACK_NH_PCT', 'VETERAN_POP_PCT', 'AGE_55_59_PCT', 'AGE_U18_PCT',
       'FAMILY_MEMBERS_UNDER_18_PCT', 'FAMILY_HH_CHILD_LT18_PCT',
       'FAMILY_HH_TOTAL', 'TOTAL_HOUSEHOLDS_PCT']


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train_df[features]
y = train_df['HOMELESS_RATE']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMRegressor(verbose=-1)

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

rmse


test_df.drop('ID', axis=1, inplace=True)
test_df[features] = test_df[features].fillna(0)
test_df = test_df[features]

predictions = model.predict(test_df)
predictions


pred_positive = []

for i in predictions:
    if i < 0:
        i = 0
    pred_positive.append(i)

sub['HOMELESS_RATE'] = pred_positive
# sub.to_csv("submission.csv", index=False)     
sub


# 複数の回帰モデルを使用して、RMSEを計算し、最適なモデルを選択する

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np  

def evaluate_models(models, X_train, y_train, X_valid, y_valid):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        results[name] = rmse
    return results


models = {
    "Linear Regression": LinearRegression(),
    'SVR': SVR(),
    "Random Forest": RandomForestRegressor(),
    "CatBoost": CatBoostRegressor(),
    "LightGBM": LGBMRegressor(verbose=-1),
    "XGBoost": XGBRegressor()
}


evaluate_models(models, X_train, y_train, X_valid, y_valid)


models = {
    "Linear Regression": LinearRegression(),
    'SVR': SVR(),
    "Random Forest": RandomForestRegressor(),
    "CatBoost": CatBoostRegressor(),
    "LightGBM": LGBMRegressor(verbose=-1),
    "XGBoost": XGBRegressor()
}


evaluate_models(models, X_train, y_train, X_valid, y_valid)


# CatboostRegressorが良さそうなので様子を見る

model_ca = CatBoostRegressor(iterations=100, learning_rate=0.2, depth=8, verbose=0)
model_ca.fit(X_train, y_train)
y_pred_ca = model_ca.predict(X_valid)
rmse_ca = np.sqrt(mean_squared_error(y_valid, y_pred_ca))
rmse_ca


# catboost regressorのハイパーパラメータをグリッドサーチして、RMSEを最小化する
from sklearn.model_selection import GridSearchCV

params = {
    'iterations': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [4, 6, 8]
}

grid_search = GridSearchCV(estimator=CatBoostRegressor(), param_grid=params, scoring='neg_root_mean_squared_error', cv=3)

best_params = grid_search.fit(X_train, y_train).best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_valid)
RMSE_CatBoost = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Best parameters: {best_params}")
print(f"RMSE: {RMSE_CatBoost}")




prediction = best_model.predict(test_df[features])
prediction




pred_positive = []

for i in predictions:
    if i < 0:
        i = 0
    pred_positive.append(i)

sub['HOMELESS_RATE'] = pred_positive
sub.to_csv("submission.csv", index=False)     
sub










