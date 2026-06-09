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


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# Carregar dados
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Explorando os dados
print("Shape do conjunto de treino:", train.shape)
print("Shape do conjunto de teste:", test.shape)
print("Descrição estatística dos dados de treino:")
print(train.describe())


#verificando se hávalores vazios nos dataset de treino
train.isna().sum()


#verificando se hávalores vazios nos dataset de teste
test.isna().sum()


#preenchendo de forma arbitrária (igual a 50) o valor faltante para "winddirection"
test.fillna(50,inplace=True)


# Removendo colunas irrelevantes para o modelo ('id' e 'day')
features_to_drop = ['id', 'day']
X_train = train.drop(columns=['rainfall'] + features_to_drop)
X_test = test.drop(columns=features_to_drop)
y_train = train['rainfall']


# Normalizando os dados para melhorar a performance do modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Treinando o modelo Random Forest 
model = RandomForestClassifier(
    n_estimators=100,  
    max_depth=8, 
    min_samples_split=10, 
    min_samples_leaf=4,  
    max_features='log2',
    class_weight='balanced_subsample',
    random_state=42
)
model.fit(X_train, y_train)


#coletando as predições no conjunto de treinamento
pred_train = model.predict(X_train)
pred_train


# Avaliação do modelo no conjunto de treinamento utilizando a métrica AUC-ROC
pred_train = model.predict(X_train)
auc_score = roc_auc_score(y_train, pred_train)
print(f"AUC no conjunto de treinamento: {auc_score:.4f}")


#coletando as predições no conjunto de teste (para submissão)
pred_test = model.predict(X_test)
pred_test


#preparando a submissão
submission['rainfall'] = pred_test
submission


#salvando o arquivo para submissão (sem índice para ficar igual o formato definido pela competição)

submission.to_csv('sub_01.csv', index=False)

