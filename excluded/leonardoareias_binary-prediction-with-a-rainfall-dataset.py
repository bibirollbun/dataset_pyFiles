import pandas as pd
import numpy as np


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()


train.isna().sum()


test.isna().sum()


train.describe()


for column in train.columns:
    if column not in ["id", "day", "rainfall"]:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=train["rainfall"], y=train[column])
        plt.title(f"{column} vs. Rainfall")
        plt.xlabel("Rainfall (0 = No, 1 = Yes)")
        plt.ylabel(column)
        plt.show()



# Identificando a quantidade mÃ­nima de nuvens em dias de chuva
train.loc[train['rainfall'] == 1, 'cloud'].min()


# Identificando a Ãºmidade mÃ­nima de nuvens em dias de chuva
train.loc[train['rainfall'] == 1, 'humidity'].min()


def identifica_condicoes(data, alvo, valor_alvo):
    for i in data:
        condicao_i = data.loc[data[alvo] == valor_alvo, i].min()
        print(f'Valor mÃ­nimo de {i} em dias de chuva {condicao_i}')


print(identifica_condicoes(train, 'rainfall', 1))


train = train[train["humidity"] >= 58.0]
train = train[train['cloud'] >= 20.0]


test[test.isnull().any(axis=1)]


coluna_nula = "winddirection"  # Substitua pelo nome da coluna com o valor ausente
test[coluna_nula].fillna(test[coluna_nula].mean(), inplace=True)


# Confirmando se ainda hÃ¡ nulos
test.isna().sum()


# 1. Identificar se a base estÃ¡ desbalanceada
train['rainfall'].value_counts()


# 2. Escolher o modelo a ser utilizado (RegressÃ£o, ClassificaÃ§Ã£o ou Agrupamento)
from sklearn.linear_model import LogisticRegression


# 3. PadronizaÃ§Ã£o dos dados
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Criar pipeline com padronizaÃ§Ã£o e regressÃ£o logÃ­stica
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Padroniza os dados
    ('model', LogisticRegression())  # Modelo de regressÃ£o logÃ­stica
])


# Separar as features (X) e o alvo (y) no conjunto de treino
X_train = train.drop(columns=['rainfall', 'id']) 
y_train = train['rainfall']  


# Treinar o modelo com os dados de treino
pipeline.fit(X_train, y_train)


# Aplicar o modelo ao conjunto de teste
X_test = test.drop(columns=['id'])  
test['rainfall'] = pipeline.predict(X_test)  


# Salvar o arquivo CSV com as previsÃµes
test[['id', 'rainfall']].to_csv("submission.csv", index=False)


resultado = pd.read_csv('/kaggle/working/submission.csv')


resultado.loc[resultado['id'] == 2707]




