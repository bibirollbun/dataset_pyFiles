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


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


dados = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
dados.describe()
dados_testes = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submissao = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')



y = dados['rainfall']
X = dados.drop(['rainfall','id','day'],axis='columns')



model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=15, min_samples_leaf=8, random_state=25).fit(X,y)


pred = model.predict(X)
pred


model.score(X,y)


from sklearn.metrics import roc_auc_score
roc_auc_score(y,pred)


dados_testes.drop(['id','day'],axis='columns', inplace=True)
dados_testes['winddirection']
dados_testes.fillna(50, inplace=True)



pred_final = model.predict(dados_testes)


pred_final


submissao['rainfall'] = pred_final
submissao



submissao.to_csv('sub1.csv',index=False)

