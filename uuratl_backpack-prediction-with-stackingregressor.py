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

import xgboost as xg 
import lightgbm as lgm
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import StackingRegressor
import optuna
import matplotlib.pyplot as plt
import seaborn as sb
import xgboost as xg 
import lightgbm as lgm
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")


base = '/kaggle/input/playground-series-s5e2/'
test_df = pd.read_csv(base + "test.csv")
train_df = pd.read_csv(base + "train.csv")


train_df.head()


test_df.head()


train_df.info(), test_df.info()


train_df.describe()


test_df.describe()


cols = ['Brand', 'Material', 'Size', 'Style']
for col in cols:
    print('Unique', col, train_df[col].unique())


train_df.isna().sum()


test_df.isna().sum()


train_df[['Material', 'Style', 'Size', 'Weight Capacity (kg)']].groupby(['Style', 'Material', 'Size' ]).agg(WeightCapacityAVG=('Weight Capacity (kg)', 'mean'),
                                                                                      WeightCapacityMax=('Weight Capacity (kg)', 'max'),
                                                                                       WeightCapacityMin=('Weight Capacity (kg)', 'min'))


colors = train_df['Color'].unique()[:-1]
color_palette = dict(zip(colors, colors))


train['Weight Capacity (kg)'].hist();


train['Compartments'].hist();


plt.figure(figsize=(20,4));
plt.subplot(1,4,1);
sb.boxplot(x='Brand', y='Price', data=train_df);
plt.title('Brand vs Price', fontsize=10);
plt.xlabel('Brand', fontsize=9);
plt.xticks(rotation=30)
plt.ylabel('Price (USD)', fontsize=9);

plt.subplot(1,4,2)

sb.boxplot(x='Style', y='Price', data=train_df);
plt.title('Style vs Price', fontsize=10);
plt.xlabel('Style', fontsize=9);
plt.xticks(rotation=30)
plt.ylabel('Price (USD)', fontsize=9);

plt.subplot(1,4,3)

sb.boxplot(x='Material', y='Price', data=train_df);
plt.title('Material vs Price', fontsize=10);
plt.xlabel('Material', fontsize=9);
plt.xticks(rotation=30)
plt.ylabel('Price (USD)', fontsize=9);

plt.subplot(1,4,4)

sb.boxplot(x='Color', y='Price', data=train_df, palette=color_palette);
plt.title('Color vs Price', fontsize=10);
plt.xlabel('Color', fontsize=9);
plt.xticks(rotation=30)
plt.ylabel('Price (USD)', fontsize=9);


plt.figure(figsize=(15,3));
plt.subplot(1,2,1);
sb.boxplot(x='Material', y='Weight Capacity (kg)', data=train_df);
plt.title('Material vs Weight Capacity (kg)', fontsize=10);
plt.xlabel('Material', fontsize=9);
plt.xticks(rotation=30)
plt.ylabel('Weight Capacity (kg)', fontsize=9);

plt.subplot(1,2,2)

sb.boxplot(x='Size', y='Weight Capacity (kg)', data=train_df);
plt.title('Size vs Weight Capacity (kg)', fontsize=10);
plt.xlabel('Size', fontsize=9);
plt.xticks(rotation=30)
plt.ylabel('Weight Capacity (kg)', fontsize=9);


plt.figure(figsize=(15,3));

plt.subplot(1,4,1);
plt.pie(train_df['Brand'].value_counts().values, labels=train_df['Brand'].value_counts().index,autopct='%.0f%%') 
plt.title('Distribution of Brands', fontsize=10);

plt.subplot(1,4,2);
plt.pie(train_df['Material'].value_counts().values, labels=train_df['Material'].value_counts().index,autopct='%.0f%%')
plt.title('Distribution of Materials', fontsize=10);

plt.subplot(1,4,3);
plt.pie(train_df['Size'].value_counts().values, labels=train_df['Size'].value_counts().index,autopct='%.0f%%')
plt.title('Distribution of Size', fontsize=10);

plt.subplot(1,4,4);
plt.pie(train_df['Style'].value_counts().values, labels=train_df['Style'].value_counts().index,autopct='%.0f%%')
plt.title('Distribution of Style', fontsize=10);

plt.show() 


def fill_na(data):
    df = data.copy()
    ## Filling nan brands with most popular one.
    df["Brand"] = train_df["Brand"].fillna(df["Brand"].mode()[0])
    ## Brands generally make the same style of bags, I filled nans with most popular style of brands.
    df["Style"] = df[["Brand", "Style"]].groupby("Brand", sort=False)["Style"].transform(lambda x: x.fillna(x.mode()[0])).values
    
    df["Material"] = df[["Brand", "Style", "Material"]].groupby(["Brand", "Style"], sort=False)["Material"].transform(lambda x: x.fillna(x.mode()[0])).values

    df["Size"] = df[["Brand", "Style", "Material", "Size", 'Compartments']].groupby(["Brand", "Style", "Material", 'Compartments'], sort=False)["Size"].transform(lambda x: x.fillna(x.mode()[0])).values

    df["Waterproof"] = df[["Style", "Material", "Waterproof"]].groupby(["Style", "Material"], sort=False)["Waterproof"].transform(lambda x: x.fillna(x.mode()[0])).values

    df["Weight Capacity (kg)"] = df[["Style", "Material", "Size", "Weight Capacity (kg)"]].groupby(["Style", "Material", "Size"], sort=False)["Weight Capacity (kg)"].transform(lambda x: x.fillna(x.median())).values

    df["Color"] = df[["Style", "Color", "Brand"]].groupby(["Style", "Brand"], sort=False)["Color"].transform(lambda x: x.fillna(x.mode()[0])).values


    df["Laptop Compartment"] = df[["Style", "Size", "Compartments", "Laptop Compartment"]].groupby(["Style", "Size", "Compartments"], sort=False)["Laptop Compartment"].transform(lambda x: x.fillna(x.mode()[0])).values

    df['Weight Capacity Bins'] = pd.cut(df['Weight Capacity (kg)'], bins=[4.5,10,15,20,25,30.5], labels=[0, 1, 2, 3, 4]).astype('int')

    df['Compartments Bins'] = pd.cut(df['Compartments'], bins=[0,2,4,6,8,11], labels=[0, 1, 2, 3, 4]).astype('int')
    return df


def encode_df(data):
    #sc = MinMaxScaler()
    lb = lb = LabelEncoder()
    cols_c= ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
    df = data.copy()
    df[cols_c] = df[cols_c].apply(lb.fit_transform)
    return df


def prepare_data(data):
    df = data.copy()
    df = fill_na(df)
    df = encode_df(df)
    return df


test = prepare_data(test_df)
train = prepare_data(train_df)


train.isna().sum(), test.isna().sum()


test.head()


train.head()


x = train[['Brand', 'Material', 'Size', 'Compartments Bins', 'Laptop Compartment', 'Waterproof', 'Style', 'Color', 'Weight Capacity Bins']]
y = train[['Price']]


x_train, x_val, y_train, y_val = train_test_split(x, y,  test_size=0.2, random_state=42)
x_train.shape, x_val.shape, y_train.shape, y_val.shape


def cat_study(trial):
    params ={
             "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
             "depth": trial.suggest_int("depth", 3, 15),
             "subsample": trial.suggest_float("subsample", 0.05, 1.0),
             "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-6, 10.0),
             "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
             "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
             "random_state":42,
             "verbose": False} 
    
    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train)

    pred = model.predict(x_val)
    return mean_squared_error(y_val, pred, squared=False)


%%time
study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(cat_study, n_trials=20)
print("Best trial:")
print(" Value: {}".format(study_cat.best_trial.value))
print(" Params: {}".format(study_cat.best_trial.params))


def lgbr_study(trial):
    params = {'learning_rate': trial.suggest_loguniform("learning_rate", 0.01, 0.1),
              'max_depth': trial.suggest_int("max_depth", 5, 20),
              'min_child_samples' : trial.suggest_int("min_child_samples", 5, 50),
              'subsample': trial.suggest_float("subsample", 0.5, 1.0),
              'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
              'n_estimators': trial.suggest_int("n_estimators", 500, 5000),
              'random_state': 42,
              'verbose': -1,
              'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 10.0),
              'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 10.0)}
    
    model = lgm.LGBMRegressor(**params)
    model.fit(x_train, y_train)

    pred = model.predict(x_val)
    return mean_squared_error(y_val, pred, squared=False)


%%time
study_lgbr = optuna.create_study(direction="minimize")
study_lgbr.optimize(lgbr_study, n_trials=20)
print("Best trial:")
print(" Value: {}".format(study_lgbr.best_trial.value))
print(" Params: {}".format(study_lgbr.best_trial.params))


def xgb_study(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float('subsample', 0.5, 1.0),
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0.5, 1.0),
        "gamma": trial.suggest_loguniform("gamma", 1e-6, 1e-2),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-6, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-6, 10.0),
        "random_state": 42}
    
    model_xgbr = xg.XGBRegressor(**params)
    model_xgbr.fit(x_train, y_train)

    pred = model_xgbr.predict(x_val)
    return mean_squared_error(y_val, pred, squared=False)


%%time
study_xgb= optuna.create_study(direction="minimize")
study_xgb.optimize(xgb_study, n_trials=20)
print("Best trial:")
print(" Value: {}".format(study_xgb.best_trial.value))
print(" Params: {}".format(study_xgb.best_trial.params))


base_algs = [('xgbr', xg.XGBRegressor(**study_xgb.best_trial.params)),
             ('lgbr', lgm.LGBMRegressor(**study_lgbr.best_trial.params)),
             ('cat', CatBoostRegressor(**study_cat.best_trial.params))]
meta_alg = LinearRegression()
stacking_model = StackingRegressor(estimators=base_algs, final_estimator=meta_alg, n_jobs=1)


stacking_model.fit(x,y)


predictions = stacking_model.predict(test[x.columns])


submission = pd.DataFrame({'id': test['id'],
                           'Price': predictions})

submission


submission.to_csv('submission.csv', index=False)
print("Done!")
print(submission.head())

