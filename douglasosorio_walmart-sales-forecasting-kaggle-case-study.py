# Exploratório
import numpy as np 
import pandas as pd

# Visualizações
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# Modelos
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')
features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
sample_submission = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')


train.head()


train.info()


train.describe()


stores.head()


# Check for missing values
print("\nMissing Values:")
print(train.isnull().sum())


stores.info()


features.head()


features.info()


# Explore valores únicos em variáveis categóricas
print("\nUnique Store Types:")
print(stores['Type'].unique())


print("\nUnique Departments:")
print(train['Dept'].unique())


print("\nUnique Holidays:")
print(train['IsHoliday'].unique())


# Combine informações da loja e do departamento
train['Store_Dept'] = train['Store'].astype(str) + '_' + train['Dept'].astype(str)
test['Store_Dept'] = test['Store'].astype(str) + '_' + test['Dept'].astype(str)


# Extraia o mês e o ano da coluna Data
train['Month'] = pd.to_datetime(train['Date']).dt.month
train['Year'] = pd.to_datetime(train['Date']).dt.year

test['Month'] = pd.to_datetime(test['Date']).dt.month
test['Year'] = pd.to_datetime(test['Date']).dt.year

features['Month'] = pd.to_datetime(features['Date']).dt.month
features['Year'] = pd.to_datetime(features['Date']).dt.year


# Calcular o valor total da redução de preço
features['Total_MarkDown'] = features['MarkDown1'] + features['MarkDown2'] + features['MarkDown3'] + features['MarkDown4'] + features['MarkDown5']


# Codificar variáveis categóricas
# Exemplo: Codificação one-hot para tipos de armazenamento
store_type_dummies = pd.get_dummies(stores['Type'], prefix='Store_Type', drop_first=True)
stores = pd.concat([stores, store_type_dummies], axis=1)


# Mesclar recursos adicionais aos conjuntos de dados de treinamento e teste
train = train.merge(stores, on='Store', how='left')
train = train.merge(features, on=['Store', 'Date'], how='left')

test = test.merge(stores, on='Store', how='left')
test = test.merge(features, on=['Store', 'Date'], how='left')


# Mostrar dataset atualizado
train.head()


test.head()


# Criar recursos de atraso
train['Weekly_Sales_Lag1'] = train['Weekly_Sales'].shift(1)
train['Weekly_Sales_Lag2'] = train['Weekly_Sales'].shift(2)


# Criar estatísticas contínuas
train['Rolling_Mean'] = train['Weekly_Sales'].rolling(window=4).mean()
train['Rolling_Std'] = train['Weekly_Sales'].rolling(window=4).std()


# Converter a coluna 'Date' no tipo "datetime"
train['Date'] = pd.to_datetime(train['Date'])

# Crie recursos sazonais
train['Month'] = train['Date'].dt.month
train['Quarter'] = train['Date'].dt.quarter
train['WeekOfYear'] = train['Date'].dt.isocalendar().week


print(train.head())


# Converter a coluna 'Data' para o formato de data e hora
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
features['Date'] = pd.to_datetime(features['Date'])


# Classificar os conjuntos de dados por 'Data'
train = train.sort_values('Date')
test = test.sort_values('Date')
features = features.sort_values('Date')


# Defina 'Data' como índice
train.set_index('Date', inplace=True)
test.set_index('Date', inplace=True)
features.set_index('Date', inplace=True)

# Preencha os valores ausentes
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
features.fillna(0, inplace=True)

# Redefinir índice
train.reset_index(inplace=True)
test.reset_index(inplace=True)
features.reset_index(inplace=True)


features.head()


print("stores.csv columns:")
print(stores.columns)

print("\ntrain.csv columns:")
print(train.columns)

print("\nfeatures.csv columns:")
print(features.columns)


# Gráfico de contagem de tipos de lojas
plt.figure(figsize=(8, 6))
sns.countplot(data=stores, x='Type')
plt.title('Count of Store Types')
plt.xlabel('Store Type')
plt.ylabel('Count')
plt.show()


# Distribuição dos tamanhos das lojas
plt.figure(figsize=(8, 6))
sns.histplot(data=stores, x='Size', kde=True)
plt.title('Distribution of Store Sizes')
plt.xlabel('Store Size')
plt.ylabel('Count')
plt.show()


# Boxplot de tamanhos de lojas por tipo de loja
plt.figure(figsize=(8, 6))
sns.boxplot(data=stores, x='Type', y='Size')
plt.title('Store Sizes by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Store Size')
plt.show()


# Matriz de correlação
correlation_matrix = stores.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - Stores Dataset')
plt.show()


# Distribuição de vendas semanais
plt.figure(figsize=(8, 6))
sns.histplot(data=train, x='Weekly_Sales', kde=True)
plt.title('Distribution of Weekly Sales')
plt.xlabel('Weekly Sales')
plt.ylabel('Count')
plt.show()


# Boxplot de vendas semanais por loja
plt.figure(figsize=(12, 6))
sns.boxplot(data=train, x='Store', y='Weekly_Sales')
plt.title('Weekly Sales by Store')
plt.xlabel('Store')
plt.ylabel('Weekly Sales')
plt.show()


# Boxplot de vendas semanais por departamento
plt.figure(figsize=(16, 6))
sns.boxplot(data=train, x='Dept', y='Weekly_Sales')
plt.title('Weekly Sales by Department')
plt.xlabel('Department')
plt.ylabel('Weekly Sales')
plt.show()


# Gráfico de linhas de vendas semanais ao longo do tempo
plt.figure(figsize=(12, 6))
sns.lineplot(data=train, x='Date', y='Weekly_Sales')
plt.title('Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.xticks(rotation=45)
plt.show()


# Gráfico de dispersão de vendas semanais vs. temperatura
plt.figure(figsize=(8, 6))
sns.scatterplot(data=train, x='Temperature', y='Weekly_Sales')
plt.title('Weekly Sales vs. Temperature')
plt.xlabel('Temperature')
plt.ylabel('Weekly Sales')
plt.show()


# Matriz de Correlação
correlation_matrix = train.corr()

fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis'))

fig.update_layout(
    title='Correlation Matrix - Train Dataset',
    xaxis_title='Features',
    yaxis_title='Features')

# Aumentar o tamanho da figura
fig.update_layout(height=800, width=800)

# Exibir o gráfico interativo
pio.show(fig)


# Traçando a distribuição de características numéricas
numerical_cols = ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
fig, ax = plt.subplots(figsize=(15, 10))

for col in numerical_cols:
    sns.histplot(data=features, x=col, kde=True, ax=ax)
    ax.set_title("Distribution of Numerical Features")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Count")

# Traçando a distribuição de características numéricas usando Plotly
fig = px.histogram(features, x=numerical_cols, marginal="rug", nbins=30)
fig.update_layout(height=600, width=900, title="Distribution of Numerical Features")
fig.show()


# Traçando a relação entre os recursos
sns.pairplot(features[numerical_cols])
plt.show()


# Especifica a loja e departamento para previsão
store = 1
department = 1

# Filtra os dados de vendas para a loja e departamento especificados
sales_data = train[(train['Store'] == store) & (train['Dept'] == department)]['Weekly_Sales']

# Ajusta o modelo ARIMA aos dados de vendas
model = ARIMA(sales_data, order=(1, 1, 1))  # Ordem do modelo (p, d, q) - exemplo
model_fit = model.fit()

# Previsão de vendas futuras
forecast_steps = 12  # Número de períodos a serem previstos
forecast = model_fit.forecast(steps=forecast_steps)

# Imprime as vendas previstas
print(f"Previsão de vendas para Loja {store} e Departamento {department}:")
print(forecast)


# Trace as vendas previstas
plt.figure(figsize=(10, 6))
plt.plot(sales_data.index, sales_data.values, label='Historical Sales')
plt.plot(forecast.index, forecast, label='Forecasted Sales')
plt.title(f"Sales Forecast for Store {store} and Department {department}")
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# Especifica a loja e departamento para análise
store = 1
department = 1

# Filtra os dados de vendas para a loja e departamento especificados
sales_data = train[(train['Store'] == store) & (train['Dept'] == department)][['Date', 'Weekly_Sales']]

# Define a coluna Date como índice
sales_data.set_index('Date', inplace=True)

# Realiza a decomposição da série temporal
result = seasonal_decompose(sales_data['Weekly_Sales'], model='additive')


# Trace os componentes da decomposição
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(result.observed)
plt.title('Observed Sales')
plt.subplot(4, 1, 2)
plt.plot(result.trend)
plt.title('Trend')
plt.subplot(4, 1, 3)
plt.plot(result.seasonal)
plt.title('Seasonal')
plt.subplot(4, 1, 4)
plt.plot(result.resid)
plt.title('Residuals')
plt.tight_layout()
plt.show()

