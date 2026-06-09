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
import matplotlib.pyplot as plt
import seaborn as sns


# Carregar dados

train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


print("InformaÃ§Ãµes do dataset de treino:")
print(train.info())
print("\nPrimeiras linhas:")
train.head()


# Explorando os dados
print("\nShape do conjunto de treino:", train.shape)
print("\nShape do conjunto de teste:", test.shape)
print("\nShape do conjunto de submission:", submission.shape)
print("\nDescriÃ§Ã£o estatÃ­stica dos dados de treino:")
print(train.describe())


plt.figure(figsize=(10, 6))
sns.histplot(train['Listening_Time_minutes'], bins=50, kde=True)
plt.title('Tempo de Escuta')
plt.show()


missing_values = train.isnull().sum()
missing_values


train['Publication_Day'].value_counts()


episodios_por_podcast = train.groupby('Podcast_Name')['Episode_Title'].nunique().reset_index()
episodios_por_podcast.columns = ['Podcast_Name', 'Episodios_Unicos']
episodios_por_podcast


# Filtrar com mÃºltiplas condiÃ§Ãµes + resetar Ã­ndice
df = train[
    (train['Podcast_Name'] == 'Tech Trends') & 
    (train['Episode_Title'].str.contains('Episode 1')) &
    (train['Listening_Time_minutes'] > 10)
].reset_index(drop=True)

# Verificar resultados
print(f"Linhas encontradas: {len(df)}")
df.head()


# Selecionar apenas colunas numÃ©ricas
numeric_cols = train.select_dtypes(include=['float64', 'int64'])

# Calcular correlaÃ§Ã£o
correlation_matrix = numeric_cols.corr()

# Plotar mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Mapa de Calor das CorrelaÃ§Ãµes')
plt.show()



# Cria um novo DataFrame filtrando apenas os episÃ³dios que possuem duraÃ§Ã£o registrada
episodios_com_duracao = (
    train.loc[train['Episode_Length_minutes'].notna()]
    .copy()
)

# Exibe informaÃ§Ãµes sobre a amostra selecionada
print(f"ðŸ“Š Encontrados {len(episodios_com_duracao)} episÃ³dios com duraÃ§Ã£o registrada")
print("Amostra das 25 primeiras linhas:")

# ExibiÃ§Ã£o formatada
(
    episodios_com_duracao
    .head(25)
    .filter([
        'Podcast_Name', 
        'Episode_Title',
        'Episode_Length_minutes',
        'Listening_Time_minutes',
        'Release_Date'
    ])
    .style
    .format({'Episode_Length_minutes': '{:.1f} min', 
             'Listening_Time_minutes': '{:.1f} min'})
    .set_caption('EpisÃ³dios com DuraÃ§Ã£o DisponÃ­vel')
    .background_gradient(cmap='Blues', subset=['Episode_Length_minutes'])
)


episodios_com_duracao.shape


# Criar os conjuntos X e y
X = episodios_com_duracao['Episode_Length_minutes'].copy()
y = episodios_com_duracao['Listening_Time_minutes'].copy()

# Amostra concisa dos dados
print("ðŸ“Œ VariÃ¡veis(X):")
print(X.head())
print("\nðŸ“Œ VariÃ¡vel(y):")
print(y.head())


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import LabelEncoder


# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 2. Modelo XGBoost BÃ¡sico
# Configurar e treinar o modelo
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,# NÃºmero de Ã¡rvores no modelo
    learning_rate=0.1,# Taxa de aprendizado (tamanho do passo em cada iteraÃ§Ã£o)
    random_state=42,
)


xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

# Fazer previsÃµes
y_pred = xgb_model.predict(X_test)

# Avaliar desempenho
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"ðŸ“Š Performance Inicial:")
print(f"RMSE: {rmse:.2f}")
print(f"RÂ²: {r2:.4f}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Tempo Real de Escuta (minutos)')
plt.ylabel('Tempo Previsto (minutos)')
plt.title('Desempenho do XGBoost - Valores Reais vs Previstos')
plt.show()


test['Episode_Length_minutes'].count()


# Identificar linhas no test_data onde Episode_Length_minutes nÃ£o Ã© numÃ©rica
# Primeiro, vamos carregar test_data como sendo igual ao train_data por enquanto (jÃ¡ que ainda nÃ£o foi fornecido)
test = test.copy()

# Converter Episode_Length_minutes para numÃ©rica (forÃ§ando erros a NaN)
test['Episode_Length_minutes'] = pd.to_numeric(test['Episode_Length_minutes'], errors='coerce')

median = test['Episode_Length_minutes'].median()

# Substituir os valores NaN (nÃ£o numÃ©ricos originalmente) pela Listening_Time_minutes correspondente
test['Episode_Length_minutes'].fillna(median, inplace=True)

# Mostrar as primeiras linhas para verificar
test[['Episode_Length_minutes']].head()


pred = xgb_model.predict(test[['Episode_Length_minutes']])

submission['Listening_Time_minutes'] = pred
submission.to_csv('sub_01.csv', index=False)

