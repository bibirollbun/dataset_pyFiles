import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os

# Modelos
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Caminho
DIR = '/kaggle/input/russian-car-plates-prices-prediction'

# Leitura dos dados
train = pd.read_csv(f'{DIR}/train.csv')
test = pd.read_csv(f'{DIR}/test.csv')
sample_submission = pd.read_csv(f'{DIR}/sample_submission.csv')
print("#####")
print(train.info())
print("#####")
print(train.describe())
print("#####")
print("Train shape:", train.shape)
print("#####")
print("Test shape:", test.shape)
print("#####")
# Verificar colunas únicas e duplicatas
print("Unique plates in train:", train['plate'].nunique())
print("#####")
print("Unique plates in test:", test['plate'].nunique())

# Verificar valores nulos
print(train.isnull().sum())

# Tradução dos códigos suplementares
print("Suplementares")
from importlib.util import spec_from_file_location, module_from_spec


def load_py_module(filepath, module_name='mod'):
    spec = spec_from_file_location(module_name, filepath)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

supplemental = load_py_module(f'{DIR}/supplemental_english.py')
region_codes = supplemental.REGION_CODES
gov_codes = supplemental.GOVERNMENT_CODES


### normalizando os valores do price 
train['price'] = np.log1p(train['price'])

train['date'] = pd.to_datetime(train['date'])
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day_of_week'] = train['date'].dt.dayofweek


# Regex para extrair partes da placa
def parse_plate(plate):
    match = re.match(r"([A-Z]{1})(\d{3})([A-Z]{2})(\d+)", plate)
    if match:
        return pd.Series(match.groups())
    else:
        return pd.Series([None]*4)

train[['first_letter', 'number', 'last_two_letters', 'region_code']] = train['plate'].apply(parse_plate)


# Inverter o dicionário
region_map = {}
for region, codes in region_codes.items():
    for code in codes:
        region_map[code] = region

train['region_name'] = train['region_code'].map(region_map)
train_encoded = pd.get_dummies(train, columns=['first_letter', 'last_two_letters', 'region_name'], drop_first=True)
X = train_encoded.drop(columns=['plate', 'price', 'date'])
y = train_encoded['price']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(random_state=42,n_estimators=10,
                              criterion='absolute_error', ### 52.8334 V10 'squared_error'
                              max_depth=None)
model.fit(X_train, y_train)
preds = model.predict(X_val)

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # evita divisão por zero
    return np.mean(diff) * 100


# Previsão no conjunto de validação
preds = model.predict(X_val)

# Reverter log1p para escala original
y_val_exp = np.expm1(y_val)
preds_exp = np.expm1(preds)

# Calcular SMAPE
smape_score = smape(y_val_exp, preds_exp)
print("SMAPE:", smape_score)

mae_score = mean_absolute_error(y_val_exp, preds_exp)
print("MAE:", mae_score)



##########################



# Repetir engenharia no test.csv
test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day_of_week'] = test['date'].dt.dayofweek
test[['first_letter', 'number', 'last_two_letters', 'region_code']] = test['plate'].apply(parse_plate)
test['region_name'] = test['region_code'].map(region_map)

# Encoding
test_encoded = pd.get_dummies(test, columns=['first_letter', 'last_two_letters', 'region_name'], drop_first=True)

# Alinhar colunas com X
test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)

# Previsão
test_preds_log = model.predict(test_encoded)
test_preds = np.expm1(test_preds_log)

# Submissão
sample_submission['price'] = test_preds
sample_submission.to_csv('submission.csv', index=False)


sns.histplot(y_val_exp, color='blue', label='Real', kde=True)
sns.histplot(preds_exp, color='red', label='Previsto', kde=True)
plt.legend()
plt.title("Distribuição Preço Real vs Previsto")
plt.xlim(0, 500_000)  # exemplo: limitar de 0 até 500 mil
plt.show()


