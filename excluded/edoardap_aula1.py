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
from sklearn.linear_model import LogisticRegression
dados.shape


y = dados['rainfall']
y


X =  dados.drop(['rainfall', 'id','day'], axis='columns')
X.shape


model = LogisticRegression(random_state=0).fit(X, y)


pred = model.predict(X)


model.score(X,y)


dados_teste = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submissao = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
dados_teste.drop(['id', 'day'], axis='columns', inplace=True)

dados_teste.fillna(50, inplace=True)
dados_teste.shape


from sklearn.metrics import roc_auc_score
roc_auc_score(y, pred)



predicao_final = model.predict(dados_teste)
predicao_final.shape


submissao['rainfall'] = predicao_final
submissao


submissao.to_csv('sub_01.csv', index=False)

