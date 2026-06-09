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


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.feature_selection import mutual_info_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import NuSVC
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


df = pd.read_csv('/kaggle/input/instant-gratification/train.csv')
df_test = pd.read_csv('/kaggle/input/instant-gratification/test.csv')


col_magic = 'wheezy-copper-turtle-magic'
indep_cols = df.columns.to_numpy().tolist()
indep_cols.remove("id")
indep_cols.remove("target")
indep_cols.remove(col_magic)
dep_col = 'target'
id_col = "id"
indep_cols_w_magic = indep_cols + [col_magic]


def train_model(df, magic, model):
    df_feat_std = df[df[col_magic] == magic][indep_cols].std().to_frame()
    df_feat_high_var = df_feat_std[df_feat_std[0] > 1.5].index.to_numpy()
    
    df_94 = df[df[col_magic] == magic]
    df_94.reset_index(inplace=True)
    pipeline = Pipeline([
        ('classifier', model)
    ])

    pipeline.fit(df_94[df_feat_high_var], df_94[dep_col])

    return (pipeline, df_feat_high_var)


def train_model_all(df, df_test, model, magic_range=range(0, 512)):
    all_res = []
    for i in magic_range:
        if (i+1) % 25 == 0:
            print (i+1)

        trained_model, features = train_model(df, i, model)

        df_test_magic = df_test[df_test[col_magic] == i]
        df_test_magic.reset_index(inplace=True)
        
        if len(df_test_magic) > 0:
            y_pred = trained_model.predict(df_test_magic[features])

            res = pd.DataFrame({
                'id': df_test_magic['id'],
                'target': y_pred
            })

            all_res.append(res)
        
    all_resp = pd.concat(all_res, ignore_index=True)

    df_test_wpred = df_test.merge(all_resp, how='left', on='id')
    return df_test_wpred[['id', col_magic, 'target']]


voting_model = VotingClassifier(estimators=[
    ('nu-svc', NuSVC(probability=True, kernel='poly')), 
    ('qda', QuadraticDiscriminantAnalysis(reg_param = 0.6))
], voting='soft')

result = train_model_all(df, df_test, voting_model, magic_range=range(0, 512))


result[['id', dep_col]].to_csv('./submission.csv', index=False)


!tail submission.csv

