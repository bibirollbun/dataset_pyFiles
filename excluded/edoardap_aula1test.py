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


dados = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
dados.shape


X =  dados.drop(['rainfall', 'id','day'], axis='columns')
X.shape


y = dados['rainfall']


import xgboost as xgb
# Criar e treinar o modelo
#model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model = xgb.XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric="logloss")

model.fit(X, y)


dados_teste = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
dados_teste.drop(['id', 'day'], axis='columns', inplace=True)

dados_teste.fillna(50, inplace=True)
# Obter as probabilidades preditas da classe positiva
y_prob = model.predict_proba(X)[:, 1]  # Probabilidades da classe 1
y_prob.shape


pred_final = model.predict_proba(dados_teste)


# Calcular os valores da Curva ROC
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
roc_auc


submissao = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submissao['rainfall'] = pred_final
submissao




submissao.to_csv('sub_01-1.csv', index=False)

