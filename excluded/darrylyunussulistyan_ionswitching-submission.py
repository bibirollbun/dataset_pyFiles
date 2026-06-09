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


df_train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
df_test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier


def l(x,low,high,dx):
    slope = (high - low) / dx
    return slope * x + low

def l0(x,low,high,dx):
    slope = (high - low) / dx
    return slope * x

def f(x,low,high,mid): return -((-low+high)/625)*(x-mid)**2+high -low

def f0(x,low,high,mid): 
    return -((-low+high)/625)*(x-mid)**2+high -low

def remove_drift(a, b, low, high, drift_fn, df = df_train, offset = 0):
    x0 = df['time'][a] - offset
    drift = drift_fn(df['time'][a:b] - x0 - offset, a, b, low, high)
    signal = df['signal'][a:b]
    return signal - drift

def remove_linear_drift(a, b, low, high, df = df_train, offset = 0):
    def lin(x, a, b, low, high):
        return l0(x, low, high, (b - a) / 10000)
    return remove_drift(a, b, low, high, lin, df, offset)

def remove_parabola_drift(a, b, low, high, df = df_train, offset = 0):
    def para(x, a, b, low, high):
        return f0(x, low, high, (b - a) / 20000)
    return remove_drift(a, b, low, high, para, df, offset)


# a.loc[10:17, 'val'] = ['11','22','33','44','55','66','77','88']
def remove_drifts(df_train_ori, df_test_ori):
    df_train = df_train_ori.copy()
    df_test = df_test_ori.copy()
    
    tmu = 10000
    # apparently, .loc is end inclusive, but select via [:] is exclusive.
    df_train.loc[50 * tmu: 60 * tmu - 1, 'signal'] = remove_linear_drift(50 * tmu, 60 * tmu, -3, 0, df_train, 0)
    df_train.loc[300 * tmu: 350 * tmu - 1, 'signal'] = remove_parabola_drift(300 * tmu, 350 * tmu, -2, 3, df_train, 0)
    df_train.loc[350 * tmu: 400 * tmu - 1, 'signal'] = remove_parabola_drift(350 * tmu, 400 * tmu, 0, 5, df_train, 0)
    df_train.loc[400 * tmu: 450 * tmu - 1, 'signal'] = remove_parabola_drift(400 * tmu, 450 * tmu, 2, 7, df_train, 0)
    df_train.loc[450 * tmu: 500 * tmu - 1, 'signal'] = remove_parabola_drift(450 * tmu, 500 * tmu, 3, 9, df_train, 0)

    tofs = 500
    df_test.loc[0 * tmu : 10 * tmu - 1, 'signal'] = remove_linear_drift(0 * tmu, 10 * tmu, -3, 0, df_test, tofs)
    df_test.loc[10 * tmu : 20 * tmu - 1, 'signal'] = remove_linear_drift(10 * tmu, 20 * tmu, 0, 3, df_test, tofs)
    df_test.loc[40 * tmu : 50 * tmu - 1, 'signal'] = remove_linear_drift(40 * tmu, 50 * tmu, -2, 1, df_test, tofs)
    df_test.loc[60 * tmu : 70 * tmu - 1, 'signal'] = remove_linear_drift(60 * tmu, 70 * tmu, 2, 5, df_test, tofs)
    df_test.loc[70 * tmu : 80 * tmu - 1, 'signal'] = remove_linear_drift(70 * tmu, 80 * tmu, 4, 7, df_test, tofs)
    df_test.loc[80 * tmu : 90 * tmu - 1, 'signal'] = remove_linear_drift(80 * tmu, 90 * tmu, -3, 0, df_test, tofs)
    df_test.loc[100 * tmu : 150 * tmu - 1, 'signal'] = remove_parabola_drift(100 * tmu, 150 * tmu, -2, 3, df_test, tofs)

    return df_train, df_test


df_train_drf, df_test_drf = remove_drifts(df_train, df_test)


def train_df(df, a, b, model):
    X_train_0 = df[['signal']][a:b]
    X_train_0.reset_index(drop=True, inplace=True)
    y_train_0 = df['open_channels'][a:b]
    y_train_0.reset_index(drop=True, inplace=True)

    model.fit(X_train_0, y_train_0)
    return model


model_01s = train_df(df_train_drf, 0, 1000000, DecisionTreeClassifier(max_depth = 1))
model_01f = train_df(df_train_drf, 1000000, 1500000, DecisionTreeClassifier(max_depth = 1))
model_03 = train_df(df_train_drf, 1500000, 2000000, DecisionTreeClassifier(max_depth = 6))
model_01cf = train_df(df_train_drf, 3000000, 3500000, DecisionTreeClassifier(max_depth = 1))
model_05 = train_df(df_train_drf, 2500000, 3000000, DecisionTreeClassifier(max_depth = 6))
model_210 = train_df(df_train_drf, 2000000, 2500000, DecisionTreeClassifier(max_depth = 15))


def predict(model, df, a, b):
    X_train_0 = df[['signal']][a:b]
    X_train_0.reset_index(drop=True, inplace=True)

    return model.predict(X_train_0)


df_train_drf['open_channels_pred'] = 0


def predict_test(df_test):
    tmu = 10000
    df_test.loc[0 * tmu : 10 * tmu - 1, 'open_channels_pred'] = predict(model_01s, df_test, 0 * tmu , 10 * tmu)
    df_test.loc[10 * tmu : 20 * tmu - 1, 'open_channels_pred'] = predict(model_03, df_test, 10 * tmu , 20 * tmu)
    df_test.loc[20 * tmu : 30 * tmu - 1, 'open_channels_pred'] = predict(model_05, df_test, 20 * tmu , 30 * tmu)
    df_test.loc[30 * tmu : 40 * tmu - 1, 'open_channels_pred'] = predict(model_01s, df_test, 30 * tmu , 40 * tmu)
    df_test.loc[40 * tmu : 50 * tmu - 1, 'open_channels_pred'] = predict(model_01f, df_test, 40 * tmu , 50 * tmu)
    df_test.loc[50 * tmu : 60 * tmu - 1, 'open_channels_pred'] = predict(model_210, df_test, 50 * tmu , 60 * tmu)
    df_test.loc[60 * tmu : 70 * tmu - 1, 'open_channels_pred'] = predict(model_05, df_test, 60 * tmu , 70 * tmu)
    df_test.loc[70 * tmu : 80 * tmu - 1, 'open_channels_pred'] = predict(model_210, df_test, 70 * tmu , 80 * tmu)
    df_test.loc[80 * tmu : 90 * tmu - 1, 'open_channels_pred'] = predict(model_01s, df_test, 80 * tmu , 90 * tmu)
    df_test.loc[90 * tmu : 100 * tmu - 1, 'open_channels_pred'] = predict(model_03, df_test, 90 * tmu , 100 * tmu)
    df_test.loc[100 * tmu : 150 * tmu - 1, 'open_channels_pred'] = predict(model_01cf, df_test, 100 * tmu , 150 * tmu)
    df_test.loc[150 * tmu : 200 * tmu, 'open_channels_pred'] = predict(model_01s, df_test, 150 * tmu , 200 * tmu + 1)


predict_test(df_test_drf)


df_test_drf['open_channels'] = df_test_drf['open_channels_pred'].astype(int)
df_test_drf[['time', 'open_channels']].to_csv('./submission.csv', index=False, float_format='%.4f')


!head -n 20 /kaggle/working/submission.csv

