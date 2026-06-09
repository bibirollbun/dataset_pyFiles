# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')

test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train_data.head()


test_data.head()


train_data.info()


test_data.info()


train_data.describe()


test_data.describe()


#Verificando se há dados nulos no dataset de Treino
train_data.isnull().sum()


#Verificando se há dados nulos no dataset de Teste
test_data.isnull().sum()


#Preenchendo campo 'winddirection' pela média da própria coluna

mean_winddirection = test_data["winddirection"].mean()
test_data["winddirection"] = test_data["winddirection"].fillna(mean_winddirection)
test_data.isnull().sum()


plt.figure(figsize=(12, 6))
sns.boxplot(data=train_data)
plt.xticks(rotation=45)
plt.title("Boxplot das Variáveis")
plt.show()



def remove_outliers(df):
    # Calculando o IQR para cada coluna
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Definindo os limites inferior e superior para cada coluna
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Removendo outliers
    df_clean = df[(df >= lower_bound) & (df <= upper_bound)].dropna()
    
    return df_clean


train_data_clean = remove_outliers(train_data)
train_data_clean.shape


#Identificar outliers do dataset de test

plt.figure(figsize=(12, 6))
sns.boxplot(data=test_data)
plt.xticks(rotation=45)
plt.title("Boxplot das Variáveis")
plt.show()


correlation_matrix = train_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de Correlação das Variáveis do Dataset de Treino")
plt.show()



plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix[['rainfall']].sort_values(by="rainfall", ascending=False), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlação das Variáveis com Rainfall do Dataset de Treino")
plt.show()


#Removendo colunas desnecessárias
X_train = train_data.drop(['rainfall','id','day','winddirection'], axis='columns')
X_test = test_data.drop(['id','day','winddirection'], axis='columns')

X_train.head()


#determinando a variável alvo
y_train = train_data['rainfall']
y_train.head()


from sklearn.preprocessing import StandardScaler

# Normaliza os dados (média=0, desvio=1) para melhor desempenho do modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression

# Treina o modelo de Regressão Logística com os dados normalizados
log_reg_model = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)

# Faz predições
log_reg_train_preds = log_reg_model.predict(X_train_scaled)
log_reg_train_preds


#avaliando o modelo no conjunto de treinamento (métrica: curva roc) para Regressão Logística
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train, log_reg_train_preds)


from sklearn.linear_model import LinearRegression

# Criar o modelo
lin_reg_model = LinearRegression()

# Treinar o modelo
lin_reg_model.fit(X_train_scaled, y_train)

# Fazer predições
y_pred = lin_reg_model.predict(X_train_scaled)
y_pred


#avaliando o modelo no conjunto de treinamento (métrica: curva roc) para Regressão Linear
roc_auc_score(y_train, y_pred)


from sklearn.tree import DecisionTreeRegressor

# Criar o modelo
dec_tree_model = DecisionTreeRegressor()

# Treinar o modelo
dec_tree_model.fit(X_train_scaled, y_train)

# Fazer predições
dec_tree_preds = dec_tree_model.predict(X_train_scaled)
dec_tree_preds


#avaliando o modelo no conjunto de treinamento (métrica: curva roc) para Árvore de Decisão
roc_auc_score(y_train, dec_tree_preds)


from sklearn.ensemble import RandomForestRegressor

# Criar o modelo
rand_forest_model = RandomForestRegressor()

# Treinar o modelo
rand_forest_model.fit(X_train_scaled, y_train)

# Fazer predições
rand_forest_model_preds = model.predict(X_train_scaled)
rand_forest_model_preds


#avaliando o modelo no conjunto de treinamento (métrica: curva roc) para Random Forest
roc_auc_score(y_train, rand_forest_model_preds)


from sklearn.svm import SVC

svm_model = SVC()

#Treinar Modelo 
svm_model.fit(X_train_scaled, y_train)

# Fazer previsões
svm_preds = svm_model.predict(X_train_scaled)
svm_preds


#avaliando o modelo no conjunto de treinamento (métrica: curva roc) para SVM

roc_auc_score(y_train, svm_preds)


#coletando as predições no conjunto de teste com Regressão Logistica (para submissão)
pred_test = log_reg_model.predict(X_test_scaled)
pred_test


#preparando a submissão com Regressão Logistica
submission['rainfall'] = pred_test
submission


#salvando o arquivo para submissão (sem índice para ficar igual o formato definido pela competição)

submission.to_csv('sub_rainfall.csv', index=False)

