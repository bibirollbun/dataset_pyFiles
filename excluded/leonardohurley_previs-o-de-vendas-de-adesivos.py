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


# Importar bibliotecas necessárias
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Listar arquivos na pasta de entrada
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Carregar os dados
train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Pré-processamento
train_data['date'] = pd.to_datetime(train_data['date'], errors='coerce').astype(int) / 10**9
test_data['date'] = pd.to_datetime(test_data['date'], errors='coerce').astype(int) / 10**9

train_data['num_sold'] = train_data['num_sold'].fillna(train_data['num_sold'].mean())

# Codificar variáveis categóricas (One-Hot Encoding)
train_data = pd.get_dummies(train_data, columns=['country', 'store', 'product'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['country', 'store', 'product'], drop_first=True)

# Separar variáveis independentes (X) e dependentes (y)
X = train_data.drop(columns=['num_sold', 'id'])
y = train_data['num_sold']
X_test_final = test_data.drop(columns=['id'])

# Dividir dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = GradientBoostingRegressor(max_depth=10, n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de validação
y_pred = model.predict(X_val)

# Avaliar o modelo
mae = mean_absolute_error(y_val, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Realizar cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f'Mean MAE from Cross-Validation: {-cv_scores.mean()}')

# Fazer previsões para o conjunto de teste final
y_test_pred = model.predict(X_test_final)

# Criar arquivo de submissão
submission = pd.DataFrame({'id': test_data['id'], 'num_sold': y_test_pred})
submission.to_csv('submission.csv', index=False)

# Visualizar previsões vs valores reais
plt.figure(figsize=(10, 6))
plt.plot(y_val.reset_index(drop=True), label='Real', color='blue')
plt.plot(y_pred, label='Previsto', color='red')
plt.title('Vendas Reais vs Previstas (Validação)')
plt.xlabel('Índice')
plt.ylabel('Vendas')
plt.legend()
plt.show()


