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


!pip install --pre pycaret


import pandas as pd


from pycaret.classification import *


train = pd.read_csv("../input/tabular-playground-series-apr-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-apr-2021/test.csv")


train.head(10)


train.Name.str.split(',').str[0]


def titanic_features(dataframe) :
    df = dataframe
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Sex"] = df["Sex"].map({"female":0, "male":1})
    df["Family"] = df["SibSp"]+df["Parch"]+1
    df['IsAlone'] = 0
#    df['Surname'] = df.Name.str.split(',').str[0]
    df.loc[df['Family'] == 1, 'IsAlone'] = 1
    df["Deck"] = df["Cabin"].apply(lambda x : "0" if pd.isna(x) else x[0])
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"], axis=1)    
    return df


train = titanic_features(train)
test = titanic_features(test)


train[: 2]


# To predict if Survived
setup(data=train, target='Survived', imputation_type="iterative")


get_config("X")


models = compare_models()


models


# Entrainement
gbc = create_model('gbc')


tune_gbc = tune_model(gbc)


plot_model(tune_gbc)


plot_model(tune_gbc, plot="error")


# Matrice de confusion
plot_model(tune_gbc, plot="confusion_matrix")


# Learning curve
plot_model(tune_gbc, plot="learning")


plot_model(tune_gbc, plot="feature")


plot_model(tune_gbc, plot="boundary")


plot_model(tune_gbc, plot="manifold")


catb = create_model('catboost')


tune_catb = tune_model(catb)


lgbm = create_model('lightgbm')


tune_lgbm = tune_model(lgbm)


xgb = create_model('xgboost')


tune_xgb = tune_model(xgb)


ada = create_model('ada')


tune_ada = tune_model(ada)








blended = blend_models(estimator_list = [tune_gbc, tune_catb, tune_xgb, tune_lgbm, tune_ada], 
                       fold = 5, method = 'soft')


blended = blend_models(estimator_list = [tune_gbc, tune_catb, tune_xgb, tune_lgbm, tune_ada], 
                       fold = 5, method = 'hard')


stacked = stack_models(estimator_list=[blended, tune_gbc, tune_xgb, tune_lgbm, tune_ada], 
                       meta_model=tune_catb, restack=False)


calibrated_stacked = calibrate_model(stacked)


predictions = predict_model(blended, data = test)
predictions.head()


sample_submission = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')


sample_submission['Survived'] = predictions['Label']
sample_submission.to_csv('submission.csv',index=False)






















