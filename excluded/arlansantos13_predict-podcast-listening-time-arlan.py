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


#Carregar dados

train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


#Vizualizar primeiras linhas

train_data.head()


#Verificar tipos e quantidades
train_data.info()


#Verificar detalhes

train_data.describe()


#Contabilizar dados nulos

train_data.isnull().sum()


# Verificar o número de valores únicos

train_data.nunique()


train_data.columns


train_data['Genre'].value_counts()


train_data['Episode_Sentiment'].value_counts()


train_data['Publication_Time'].value_counts()


train_data['Publication_Day'].value_counts()


import seaborn as sns
import matplotlib.pyplot as plt

x = train_data['Number_of_Ads'].dropna()

# Plot do boxplot
plt.figure(figsize=(10, 4))
sns.boxplot(x=x)

plt.title("Boxplot - Número de Anúncios por Episódio")
plt.xlabel("Número de Anúncios")
plt.show()



# Contabilizar episódios tem mais de 100 anúncios
count_ads_over_100 = train_data[train_data['Number_of_Ads'] > 100].shape[0]
print(f"Número de episódios com mais de 100 anúncios: {count_ads_over_100}")

# Visualizar as linhas com mais de 100 anúncios
train_data[train_data['Number_of_Ads'] > 100]


# Contabilizar episódios tem mais de 3 anúncios
count_ads_over_3 = train_data[train_data['Number_of_Ads'] > 3].shape[0]
print(f"Número de episódios com mais de 100 anúncios: {count_ads_over_3}")

# Visualizar as linhas com mais de 3 anúncios
train_data[train_data['Number_of_Ads'] > 3]


#Histograma da variável alvo

plt.figure(figsize=(8, 4))
plt.hist(train_data['Listening_Time_minutes'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribuição da variável alvo: Listening_Time_minutes')
plt.xlabel('Listening Time (minutos)')
plt.ylabel('Frequência')
plt.grid(True)
plt.tight_layout()
plt.show()


train_data_copy = train_data.copy()


#Criar 'Has_Guest' para indicar presença do convidado
train_data_copy['Has_Guest'] = train_data_copy['Guest_Popularity_percentage'].notnull().astype(int)

# Preenchimento de valores nulos
train_data_copy['Guest_Popularity_percentage'] = train_data_copy['Guest_Popularity_percentage'].fillna(0)
train_data_copy['Episode_Length_minutes'] = train_data_copy['Episode_Length_minutes'].fillna(train_data_copy['Episode_Length_minutes'].median())
train_data_copy['Number_of_Ads'] = train_data_copy['Number_of_Ads'].fillna(train_data_copy['Number_of_Ads'].median())


# Filtrar linhas onde Number_of_Ads <= 12
train_data_copy = train_data_copy[train_data_copy['Number_of_Ads'] <= 12]


#Contabilizar dados nulos

train_data_copy.isnull().sum()


#Conferir tamanho do df
train_data_copy.shape


# Selecionar apenas colunas numéricas
numeric_cols = train_data_copy.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = train_data_copy[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            center=0, linewidths=.5)
plt.title('Matriz de Correlação entre Variáveis Numéricas')
plt.show()


#Remover campos categóricos ("Não Importantes para o treinamento")

train_data_copy.drop(columns=['id', 'Podcast_Name', 'Episode_Title'], inplace=True)


#Separar campos categóricos

categorical_cols = ['Genre', 'Episode_Sentiment', 'Publication_Time', 'Publication_Day']


# Aplicar One-Hot Encoding

train_data_copy = pd.get_dummies(train_data_copy, columns=categorical_cols, drop_first=True)


train_data_copy.head()


#Contabilizar dados nulos

test_data.isnull().sum()


#Criar 'Has_Guest' para indicar presença do convidado
test_data['Has_Guest'] = test_data['Guest_Popularity_percentage'].notnull().astype(int)

# Preenchimento de valores nulos
test_data['Guest_Popularity_percentage'] = test_data['Guest_Popularity_percentage'].fillna(0)
test_data['Episode_Length_minutes'] = test_data['Episode_Length_minutes'].fillna(test_data['Episode_Length_minutes'].median())
test_data['Number_of_Ads'] = test_data['Number_of_Ads'].fillna(test_data['Number_of_Ads'].median())


#Remover campos categóricos ("Não Importantes para o treinamento")

test_data.drop(columns=['id', 'Podcast_Name', 'Episode_Title'], inplace=True)


# Aplicar One-Hot Encoding

test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Separar features (X) e target (y)
X = train_data_copy.drop(columns=['Listening_Time_minutes'])
y = train_data_copy['Listening_Time_minutes']

# Dividir em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
rmse_xgb = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE (XGBoost): {rmse_xgb:.4f}")


from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(n_estimators=150, learning_rate=0.1, random_state=42)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
rmse_lgbm = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE (LightGBM): {rmse_lgbm:.4f}")


test_pred = lgbm.predict(test_data)


submission['Listening_Time_minutes'] = test_pred

submission.to_csv('submission.csv', index=False)

