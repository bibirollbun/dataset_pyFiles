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

dir_path = '/kaggle/input/bi-master-24-2-deteccao-de-intrusao-de-rede/'
data = pd.read_csv(os.path.join(dir_path,'treino.csv'),index_col=0)


data.head()


data.dtypes


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

#  Seed para resultados
seed = 1
random.seed(seed)
np.random.seed(seed)


data.describe()


data.shape


data.value_counts("out")


import missingno

missingno.matrix(data, figsize=(25,5))


variancia = data.select_dtypes(include=['number']).var()
print(variancia)


from sklearn.model_selection import train_test_split

X = data.iloc[:, ~data.columns.isin(["protocol_type","flag","out"])]
y = data.out
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape,y_train.value_counts("out"))
print(y_test.shape,y_test.value_counts("out"))


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

type(X_train)


# treinar modelo

from sklearn.ensemble import RandomForestClassifier

def train(X_train, y_train, seed, min_samples_leaf=5): 
  model = RandomForestClassifier(min_samples_leaf=min_samples_leaf, random_state=seed) # tente mudar parâmetro para evitar overfitting
  model.fit(X_train, y_train);
  return model

model = train(X_train, y_train, seed)



def predict_and_evaluate(model, X_test, y_test):

  y_pred = model.predict(X_test)  # inferência do teste

  # Acurácia
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(y_test, y_pred)
  print('Acurácia: ', accuracy)

  # Matriz de confusão
  from sklearn.metrics import confusion_matrix
  confMatrix = confusion_matrix(y_pred, y_test)

  ax = plt.subplot()
  sns.heatmap(confMatrix, annot=True, fmt=".0f")
  plt.xlabel('Real')
  plt.ylabel('Previsto')
  plt.title('Matriz de Confusão')

  # Colocar os nomes
  ax.xaxis.set_ticklabels(['normal.', 'smurf.','neptune.'])
  ax.yaxis.set_ticklabels(['normal.', 'smurf.','neptune.'])
  plt.show()



predict_and_evaluate(model, X_test, y_test)


teste = pd.read_csv(os.path.join(dir_path,'teste_sem_rotulo.csv'),index_col=0)



teste.head()


teste = teste.drop(columns=["protocol_type", "flag"])
teste.head()


teste_norm = scaler.transform(teste)


teste_pred = model.predict(teste_norm)
teste_pred


teste['out'] = teste_pred


teste.head()


resultado = teste.loc[:, ["out"]]

resultado.to_csv("submissao.csv")

