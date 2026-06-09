# Importación de librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

# Convertir 'date' a formato datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Ordenar los datos por fecha dentro de cada grupo: (country, store, product)
train = train.sort_values(by=['country', 'store', 'product', 'date']).reset_index(drop=True)
test = test.sort_values(by=['country', 'store', 'product', 'date']).reset_index(drop=True)


train


test


def create_date_features(df):
    """
    Extrae características temporales de la columna 'date'.
    """
    df['year']      = df['date'].dt.year
    df['month']     = df['date'].dt.month
    df['day']       = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    return df

# Aplicar la función a train y test
train = create_date_features(train)
test  = create_date_features(test)

# Calcular agregados por grupo: (country, store, product)
group_agg = train.groupby(['country','store','product'])['num_sold']\
    .agg(['mean','median','std']).reset_index().rename(columns={'mean':'grp_mean',
                                                                 'median':'grp_median',
                                                                 'std':'grp_std'})

# Unir los agregados a los datasets de train y test
train = train.merge(group_agg, on=['country','store','product'], how='left')
test  = test.merge(group_agg, on=['country','store','product'], how='left')

# Rellenar NaN en la desviación estándar (en caso de registros únicos)
train['grp_std'] = train['grp_std'].fillna(0)
test['grp_std']  = test['grp_std'].fillna(0)


# Visualizar el comportamiento histórico de las ventas en train
# (se asume que 'train' ya fue cargado y preprocesado en celdas anteriores)

# Filtrar filas con valores reales en 'num_sold'
train_valid = train.dropna(subset=['num_sold'])

# Agrupar por fecha (podemos usar la suma total de ventas para ver la tendencia global)
sales_by_date = train_valid.groupby('date')['num_sold'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(sales_by_date['date'], sales_by_date['num_sold'], marker='o', linestyle='-')
plt.title("Evolución de Ventas (num_sold) en Train")
plt.xlabel("Fecha")
plt.ylabel("Ventas Totales")
plt.grid(True)
plt.show()



# Definir los lags y ventanas a utilizar
LAGS = [7, 28]
WINDOWS = [7, 28]

def create_lag_features(df):
    """
    Crea features de lags y rolling windows para el campo 'num_sold'.
    Se agrupa por 'country', 'store' y 'product' y se ordena por 'date'.
    """
    df = df.copy()
    df = df.sort_values(by=['country','store','product','date'])
    for lag in LAGS:
        df[f'lag_{lag}'] = df.groupby(['country','store','product'])['num_sold'].shift(lag)
    for window in WINDOWS:
        # Se utiliza shift(1) para evitar el leakage de información futura
        df[f'rolling_mean_{window}'] = df.groupby(['country','store','product'])['num_sold'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}']  = df.groupby(['country','store','product'])['num_sold'].shift(1).rolling(window).std()
    return df

# Crear las features en el conjunto de entrenamiento
train_fe = create_lag_features(train)

# Eliminar filas con valores nulos en los lags para evitar problemas de lookahead
train_fe = train_fe.dropna().reset_index(drop=True)
print("Training final después de features:", train_fe.shape)

# Definir la lista de features a utilizar para el modelo
FEATURES = [
    'year', 'month', 'day', 'dayofweek', 'is_weekend',
    'grp_mean', 'grp_median', 'grp_std'
]
# Incluir los lags y rolling windows definidos
for lag in LAGS:
    FEATURES.append(f'lag_{lag}')
for window in WINDOWS:
    FEATURES.append(f'rolling_mean_{window}')
    FEATURES.append(f'rolling_std_{window}')

print("Features:", FEATURES)


# Definir un corte temporal: los últimos 180 días se utilizan para validación
cutoff_date = train_fe['date'].max() - pd.Timedelta(days=180)
train_data = train_fe[train_fe['date'] <= cutoff_date].reset_index(drop=True)
val_data   = train_fe[train_fe['date'] > cutoff_date].reset_index(drop=True)
print("Train data:", train_data.shape, "Validation data:", val_data.shape)

# Preparar los conjuntos de features y target
X_train = train_data[FEATURES]
y_train = train_data['num_sold']
X_val   = val_data[FEATURES]
y_val   = val_data['num_sold']

# Crear datasets para LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)


def mape_lgb(preds, dtrain):
    """
    Función de evaluación para LightGBM que calcula el MAPE.
    """
    labels = dtrain.get_label()
    mape_value = np.mean(np.abs((labels - preds) / labels)) * 100
    return 'mape', mape_value, False

# Parámetros del modelo
params = {
    'objective': 'regression',
    'metric': 'rmse',  # Se muestra RMSE para monitoreo, pero se evalúa también MAPE
    'learning_rate': 0.01,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': -1
}

print("Entrenando LightGBM...")

# Callbacks: early stopping y log de evaluación cada 200 iteraciones
callbacks = [
    lgb.early_stopping(stopping_rounds=100, verbose=True),
    lgb.log_evaluation(period=200)
]

# Entrenar el modelo
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=3000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train','valid'],
    feval=mape_lgb,
    callbacks=callbacks
)

# Evaluar el modelo en el conjunto de validación
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
mape_val = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
print("MAPE en validación:", mape_val)


def forecast_group(train_grp, test_grp, model):
    """
    Realiza pronóstico iterativo para un grupo.
    
    Parámetros:
    - train_grp: DataFrame histórico para el grupo (ordenado por 'date').
    - test_grp: DataFrame de test para el grupo (ordenado por 'date').
    - model: Modelo LightGBM entrenado.
    
    Retorna:
    - Lista de predicciones para las fechas en test_grp.
    """
    # Se inicializa la historia con los datos reales
    history = train_grp[['date', 'num_sold']].copy()
    preds = []
    # Iterar sobre cada fecha única en el conjunto de test (orden cronológico)
    for cur_date in test_grp['date'].sort_values().unique():
        rec = {}
        rec['date'] = cur_date
        rec['year'] = cur_date.year
        rec['month'] = cur_date.month
        rec['day'] = cur_date.day
        rec['dayofweek'] = cur_date.dayofweek
        rec['is_weekend'] = int(cur_date.dayofweek in [5,6])
        # Agregados por grupo (constantes para el grupo)
        rec['grp_mean'] = train_grp['num_sold'].mean()
        rec['grp_median'] = train_grp['num_sold'].median()
        rec['grp_std'] = train_grp['num_sold'].std() if train_grp['num_sold'].std() > 0 else 0
        # Calcular lags: si no hay suficientes datos, se usa el promedio del grupo
        for lag in LAGS:
            if len(history) >= lag:
                rec[f'lag_{lag}'] = history.iloc[-lag]['num_sold']
            else:
                rec[f'lag_{lag}'] = rec['grp_mean']
        # Calcular rolling windows: media y desviación en las últimas 'window' observaciones
        for window in WINDOWS:
            if len(history) >= window:
                rec[f'rolling_mean_{window}'] = history['num_sold'].iloc[-window:].mean()
                rec[f'rolling_std_{window}'] = history['num_sold'].iloc[-window:].std()
            else:
                rec[f'rolling_mean_{window}'] = rec['grp_mean']
                rec[f'rolling_std_{window}'] = 0
        # Crear un DataFrame temporal con las features calculadas
        df_rec = pd.DataFrame([rec])
        X_rec = df_rec[FEATURES]
        # Realizar la predicción para la fecha actual
        pred = model.predict(X_rec, num_iteration=model.best_iteration)[0]
        preds.append(pred)
        # Actualizar la historia agregando la nueva predicción usando pd.concat
        history = pd.concat([history, pd.DataFrame([{'date': cur_date, 'num_sold': pred}])], ignore_index=True)
    return preds

# Aplicar forecasting iterativo para cada grupo y asignar las predicciones en el dataset de test
test['pred_num_sold'] = np.nan
group_keys = ['country','store','product']
for key, test_grp in tqdm(test.groupby(group_keys), desc="Forecasting por grupo"):
    cond = (train['country'] == key[0]) & (train['store'] == key[1]) & (train['product'] == key[2])
    train_grp = train[cond].sort_values(by='date').copy()
    test_grp = test_grp.sort_values(by='date').copy()
    group_preds = forecast_group(train_grp, test_grp, model)
    test.loc[test.index.isin(test_grp.index), 'pred_num_sold'] = group_preds

# Verificar que no existan filas sin predicción
assert test['pred_num_sold'].isnull().sum() == 0, "Faltan predicciones en test"


# El archivo de submission debe tener las columnas "id" y "num_sold"
submission['num_sold'] = test['pred_num_sold']
submission.to_csv('submission.csv', index=False)
print("Submission generado: submission.csv")


plt.figure(figsize=(10,6))
sns.histplot(submission['num_sold'], bins=50, kde=True)
plt.title("Distribución de Predicciones en Test")
plt.xlabel("num_sold")
plt.show()


# Visualizar los pronósticos de ventas en test
# (se asume que 'test' ya contiene la columna 'pred_num_sold' con las predicciones)

# Agrupar por fecha: sumatoria de ventas pronosticadas por día
forecast_by_date = test.groupby('date')['pred_num_sold'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(forecast_by_date['date'], forecast_by_date['pred_num_sold'], marker='o', linestyle='-')
plt.title("Pronóstico de Ventas (pred_num_sold) en Test")
plt.xlabel("Fecha")
plt.ylabel("Ventas Pronosticadas")
plt.grid(True)
plt.show()


