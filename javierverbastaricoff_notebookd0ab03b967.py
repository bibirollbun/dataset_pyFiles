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
from sklearn.linear_model import LinearRegression

# 1. Cargar datos
train_set = pd.read_csv('/kaggle/input/price-house-prediction-2024-b-posgraduate-dsub/train_set.csv',index_col=0)
test_set = pd.read_csv('/kaggle/input/price-house-prediction-2024-b-posgraduate-dsub/test_set.csv',index_col=0)

# =========================
# 2. IMPUTACIÓN EN TRAIN
# =========================

# A. Imputar BuildingArea en TRAIN
train_set['Rooms_Bathroom'] = train_set['Rooms'].astype(str) + 'R_' + train_set['Bathroom'].astype(str) + 'B'
building_medians = train_set.groupby('Rooms_Bathroom')['BuildingArea'].median()

train_set['BuildingArea_imputed'] = train_set.apply(
    lambda row: building_medians.get(row['Rooms_Bathroom'], np.nan) if pd.isna(row['BuildingArea']) else row['BuildingArea'],
    axis=1
)

# Para los casos no cubiertos, usar mediana por Rooms
room_medians = train_set.groupby('Rooms')['BuildingArea'].median()
train_set['BuildingArea_imputed'] = train_set.apply(
    lambda row: room_medians.get(row['Rooms'], np.nan) if pd.isna(row['BuildingArea_imputed']) else row['BuildingArea_imputed'],
    axis=1
)

# B. Imputar YearBuilt en TRAIN
year_medians = train_set.groupby(['Suburb', 'Type'])['YearBuilt'].median()
type_medians = train_set.groupby('Type')['YearBuilt'].median()

train_set['YearBuilt_imputed'] = train_set.apply(
    lambda row: year_medians.get((row['Suburb'], row['Type']), np.nan) if pd.isna(row['YearBuilt']) else row['YearBuilt'],
    axis=1
)

train_set['YearBuilt_imputed'] = train_set.apply(
    lambda row: type_medians.get(row['Type'], np.nan) if pd.isna(row['YearBuilt_imputed']) else row['YearBuilt_imputed'],
    axis=1
)

# =========================
# 3. ENTRENAR EL MODELO
# =========================

features = ['Rooms', 'Landsize', 'BuildingArea_imputed', 'YearBuilt_imputed']
X_train = train_set[features]
y_train = train_set[['Price']]

model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 4. PREPARAR EL TEST SET IGUAL QUE EL TRAIN
# =========================

# Imputar BuildingArea
test_set['Rooms_Bathroom'] = test_set['Rooms'].astype(str) + 'R_' + test_set['Bathroom'].astype(str) + 'B'
test_set['BuildingArea_imputed'] = test_set.apply(
    lambda row: building_medians.get(row['Rooms_Bathroom'], np.nan) if pd.isna(row['BuildingArea']) else row['BuildingArea'],
    axis=1
)

test_set['BuildingArea_imputed'] = test_set.apply(
    lambda row: room_medians.get(row['Rooms'], np.nan) if pd.isna(row['BuildingArea_imputed']) else row['BuildingArea_imputed'],
    axis=1
)
# Para BuildingArea: usar mediana por Rooms si faltan datos
test_set['BuildingArea_imputed'] = test_set.apply(
    lambda row: room_medians.get(row['Rooms'], np.nan) if pd.isna(row['BuildingArea_imputed']) else row['BuildingArea_imputed'],
    axis=1
)


# Imputar YearBuilt
test_set['YearBuilt_imputed'] = test_set.apply(
    lambda row: year_medians.get((row['Suburb'], row['Type']), np.nan) if pd.isna(row['YearBuilt']) else row['YearBuilt'],
    axis=1
)
# Para YearBuilt: usar mediana por Type si faltan datos
test_set['YearBuilt_imputed'] = test_set.apply(
    lambda row: type_medians.get(row['Type'], np.nan) if pd.isna(row['YearBuilt_imputed']) else row['YearBuilt_imputed'],
    axis=1
)

test_set['YearBuilt_imputed'] = test_set.apply(
    lambda row: type_medians.get(row['Type'], np.nan) if pd.isna(row['YearBuilt_imputed']) else row['YearBuilt_imputed'],
    axis=1
)

# =========================
# 5. PREDICCIÓN Y ENVÍO
# =========================

X_test = test_set[features]

X_test = X_test.fillna(0)

y_pred = model.predict(X_test)

# Crear archivo de submission

# df_output = pd.DataFrame({'Price': y_pred.flatten()})
# df_output = df_output.reset_index()
# df_output.columns = ['index', 'Price']
# df_output.to_csv('submission.csv', index=False)



from sklearn.metrics import mean_squared_error, r2_score

# Evaluación en el train_set
y_pred_train = model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2 = r2_score(y_train, y_pred_train)

print("----- EVALUACIÓN EN TRAIN SET ------")
print("RMSE:", rmse)
print("R²:", r2)


# Copiamos dataset original
df = train_set.copy()

# Qué suburbios tienen datos faltantes en 'Car'?
missing_car = df[df['Car'].isna()]
suburbs_with_nan_car = missing_car['Suburb'].unique()

# Calculamos la mediana de 'Car' por 'Suburb'
car_median_by_suburb = df.groupby('Suburb')['Car'].median()

# Imputamos los valores faltantes con la mediana correspondiente de su 'Suburb'
df['Car'] = df.apply(
    lambda row: car_median_by_suburb[row['Suburb']] if pd.isna(row['Car']) else row['Car'],
    axis=1
)

# Verificamos si aún quedan NaNs en Car
remaining_nans_car = df['Car'].isna().sum()

missing_car

suburbs_with_nan_car

car_median_by_suburb

remaining_nans_car



# Copiamos nuevamente para mantener el control del proceso
df_model = df.copy()

# Paso A: Imputar BuildingArea
# Usamos la mediana agrupada por combinación de Rooms y Bathroom
building_medians = df_model.groupby(['Rooms', 'Bathroom'])['BuildingArea'].median()

# Para valores faltantes, intentamos imputar con esa combinación
def impute_building_area(row):
    if pd.isna(row['BuildingArea']):
        key = (row['Rooms'], row['Bathroom'])
        if key in building_medians:
            return building_medians[key]
        else:
            return None
    else:
        return row['BuildingArea']

df_model['BuildingArea'] = df_model.apply(impute_building_area, axis=1)

# Para los que aún queden, usamos la mediana por Rooms
room_medians = df_model.groupby('Rooms')['BuildingArea'].median()
df_model['BuildingArea'] = df_model.apply(
    lambda row: room_medians[row['Rooms']] if pd.isna(row['BuildingArea']) else row['BuildingArea'],
    axis=1
)

# Paso B: Imputar YearBuilt
# Usamos la mediana por combinación Suburb + Type
year_medians = df_model.groupby(['Suburb', 'Type'])['YearBuilt'].median()

def impute_year_built(row):
    if pd.isna(row['YearBuilt']):
        key = (row['Suburb'], row['Type'])
        if key in year_medians:
            return year_medians[key]
        else:
            return None
    else:
        return row['YearBuilt']

df_model['YearBuilt'] = df_model.apply(impute_year_built, axis=1)

# Para valores residuales, usamos la mediana por Type
type_medians = df_model.groupby('Type')['YearBuilt'].median()
df_model['YearBuilt'] = df_model.apply(
    lambda row: type_medians[row['Type']] if pd.isna(row['YearBuilt']) else row['YearBuilt'],
    axis=1
)

# Verificamos si aún quedan NaNs en las columnas imputadas
remaining_nans = df_model[['BuildingArea', 'YearBuilt']].isna().sum()
remaining_nans



remaining_nans = df_model.isna().sum()
remaining_nans


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Selección de variables numéricas estratégicas (sin NaNs y sin transformaciones)
features = [
    'Rooms', 'Bedroom2', 'Bathroom', 'Car',
    'Landsize', 'BuildingArea', 'YearBuilt',
    'Propertycount', 'Distance'
]

# Definimos X e y
X = df_model[features]
y = df_model['Price']

# División en conjunto de entrenamiento y validación (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificamos las formas para asegurarnos de que todo esté correcto
X_train.shape, X_val.shape, y_train.shape, y_val.shape

# Instanciamos el modelo de regresión lineal
model = LinearRegression()

# Entrenamos el modelo con el conjunto de entrenamiento
model.fit(X_train, y_train)

# Realizamos predicciones sobre el conjunto de validación
y_pred = model.predict(X_val)

# Calculamos métricas de evaluación
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

rmse, r2


# Definimos las variables de entrada y salida sobre TODO el train_set
X_full = df_model[features]
y_full = df_model['Price']

# Hacemos predicciones sobre ese mismo conjunto
y_pred_full = model.predict(X_full)

# Calculamos métricas sobre TODO el conjunto
rmse_full = np.sqrt(mean_squared_error(y_full, y_pred_full))
r2_full = r2_score(y_full, y_pred_full)

rmse_full, r2_full



# =========================
# 1. IMPORTAR LIBRERÍAS
# =========================
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# =========================
# 2. CARGAR LOS DATOS
# =========================
train_set = pd.read_csv('/kaggle/input/price-house-prediction-2024-b-posgraduate-dsub/train_set.csv', index_col=0)
test_set = pd.read_csv('/kaggle/input/price-house-prediction-2024-b-posgraduate-dsub/test_set.csv', index_col=0)

# =========================
# 3. PREPROCESAMIENTO TRAIN
# =========================
train = train_set.copy()

# Filtrado de outliers
train = train[
    (train["Rooms"] <= 7) &
    (train["Bathroom"] <= 5) &
    (train["Bedroom2"] <= 6) &
    (train["Car"] <= 6)
]
train = train[train["Landsize"] <= 20000]
df_nan = train[train["BuildingArea"].isna()]
df_not_nan = train[train["BuildingArea"].notna()]
df_not_nan = df_not_nan[df_not_nan["BuildingArea"] <= 1000]
train = pd.concat([df_not_nan, df_nan], ignore_index=True)

# Crear Rooms_Bathroom
train['Rooms_Bathroom'] = train['Rooms'].astype(str) + 'R_' + train['Bathroom'].astype(str) + 'B'

# Imputar BuildingArea
building_medians_rb = train.groupby('Rooms_Bathroom')['BuildingArea'].median()
train['BuildingArea_imputed'] = train.apply(
    lambda row: building_medians_rb.get(row['Rooms_Bathroom'], np.nan) if pd.isna(row['BuildingArea']) else row['BuildingArea'],
    axis=1
)
room_medians = train.groupby('Rooms')['BuildingArea'].median()
train['BuildingArea_imputed'] = train.apply(
    lambda row: room_medians.get(row['Rooms'], np.nan) if pd.isna(row['BuildingArea_imputed']) else row['BuildingArea_imputed'],
    axis=1
)

# Imputar YearBuilt → Age
year_medians_st = train.groupby(['Suburb', 'Type'])['YearBuilt'].median()
type_medians = train.groupby('Type')['YearBuilt'].median()
train['YearBuilt_imputed'] = train.apply(
    lambda row: year_medians_st.get((row['Suburb'], row['Type']), np.nan) if pd.isna(row['YearBuilt']) else row['YearBuilt'],
    axis=1
)
train['YearBuilt_imputed'] = train.apply(
    lambda row: type_medians.get(row['Type'], np.nan) if pd.isna(row['YearBuilt_imputed']) else row['YearBuilt_imputed'],
    axis=1
)
train['Age'] = 2024 - train['YearBuilt_imputed']
train['Age'] = train['Age'].replace(0, np.nan)
train['Age'] = train['Age'].fillna(train['Age'].median())

# Imputar Landsize
train['Landsize'] = train['Landsize'].replace(0, np.nan)
lands_medians = train.groupby('Rooms_Bathroom')['Landsize'].median()
train['Landsize'] = train.apply(
    lambda row: lands_medians.get(row['Rooms_Bathroom'], row['Landsize']) if pd.isna(row['Landsize']) else row['Landsize'],
    axis=1
)

# Transformar Distance
train['sqrt_Distance'] = np.sqrt(train['Distance'])

# Variables del modelo
features = ['Rooms', 'Bathroom', 'Car', 'Landsize', 'BuildingArea_imputed', 'Age', 'sqrt_Distance']
X_train = train[features]
y_train = train["Price"]

# =========================
# 4. ENTRENAR LASSO(alpha=100)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

lasso = Lasso(alpha=100)
lasso.fit(X_train_scaled, y_train)

# =========================
# 5. PREPROCESAMIENTO TEST
# =========================
test = test_set.copy()
test['Rooms_Bathroom'] = test['Rooms'].astype(str) + 'R_' + test['Bathroom'].astype(str) + 'B'

# Imputar BuildingArea
test['BuildingArea_imputed'] = test.apply(
    lambda row: building_medians_rb.get(row['Rooms_Bathroom'], np.nan) if pd.isna(row['BuildingArea']) else row['BuildingArea'],
    axis=1
)
test['BuildingArea_imputed'] = test.apply(
    lambda row: room_medians.get(row['Rooms'], np.nan) if pd.isna(row['BuildingArea_imputed']) else row['BuildingArea_imputed'],
    axis=1
)

# Imputar YearBuilt → Age
test['YearBuilt_imputed'] = test.apply(
    lambda row: year_medians_st.get((row['Suburb'], row['Type']), np.nan) if pd.isna(row['YearBuilt']) else row['YearBuilt'],
    axis=1
)
test['YearBuilt_imputed'] = test.apply(
    lambda row: type_medians.get(row['Type'], np.nan) if pd.isna(row['YearBuilt_imputed']) else row['YearBuilt_imputed'],
    axis=1
)
test['Age'] = 2024 - test['YearBuilt_imputed']
test['Age'] = test['Age'].replace(0, np.nan)
test['Age'] = test['Age'].fillna(train['Age'].median())

# Imputar Landsize
test['Landsize'] = test['Landsize'].replace(0, np.nan)
test['Landsize'] = test.apply(
    lambda row: lands_medians.get(row['Rooms_Bathroom'], row['Landsize']) if pd.isna(row['Landsize']) else row['Landsize'],
    axis=1
)

# Transformar Distance
test['sqrt_Distance'] = np.sqrt(test['Distance'])

# =========================
# 6. PREDICCIÓN Y ENVÍO
# =========================
X_test = test[features]
X_test = X_test.fillna(0)
X_test_scaled = scaler.transform(X_test)

y_pred = lasso.predict(X_test_scaled)

# df_output = pd.DataFrame({'Price': y_pred})
# df_output = df_output.reset_index()
# df_output.columns = ['index', 'Price']
# df_output.to_csv('submission.csv', index=False)


# =========================
# 1. IMPORTAR LIBRERÍAS
# =========================
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# =========================
# 2. CARGAR LOS DATOS
# =========================
train_set = pd.read_csv('/kaggle/input/price-house-prediction-2024-b-posgraduate-dsub/train_set.csv', index_col=0)
test_set = pd.read_csv('/kaggle/input/price-house-prediction-2024-b-posgraduate-dsub/test_set.csv', index_col=0)

# =========================
# 3. FILTRAR OUTLIERS (IQR)
# =========================
df = train_set.copy()
variables = ["Price", "Rooms", "Bathroom", "Bedroom2", "Car", "Landsize", "BuildingArea", "Distance"]
mask = pd.Series(True, index=df.index)

for var in variables:
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask &= (df[var].isna() | ((df[var] >= lower) & (df[var] <= upper)))

train = df[mask].copy()

# =========================
# 4. IMPUTACIÓN JERÁRQUICA
# =========================

# Rooms_Bathroom
train['Rooms_Bathroom'] = train['Rooms'].astype(str) + 'R_' + train['Bathroom'].astype(str) + 'B'

# BuildingArea imputación
building_medians_rb = train.groupby('Rooms_Bathroom')['BuildingArea'].median()
room_medians = train.groupby('Rooms')['BuildingArea'].median()

train['BuildingArea_imputed'] = train.apply(
    lambda row: building_medians_rb.get(row['Rooms_Bathroom'], np.nan) if pd.isna(row['BuildingArea']) else row['BuildingArea'],
    axis=1
)
train['BuildingArea_imputed'] = train.apply(
    lambda row: room_medians.get(row['Rooms'], np.nan) if pd.isna(row['BuildingArea_imputed']) else row['BuildingArea_imputed'],
    axis=1
)

# YearBuilt imputación → Age
year_medians_st = train.groupby(['Suburb', 'Type'])['YearBuilt'].median()
type_medians = train.groupby('Type')['YearBuilt'].median()

train['YearBuilt_imputed'] = train.apply(
    lambda row: year_medians_st.get((row['Suburb'], row['Type']), np.nan) if pd.isna(row['YearBuilt']) else row['YearBuilt'],
    axis=1
)
train['YearBuilt_imputed'] = train.apply(
    lambda row: type_medians.get(row['Type'], np.nan) if pd.isna(row['YearBuilt_imputed']) else row['YearBuilt_imputed'],
    axis=1
)
train['Age'] = 2024 - train['YearBuilt_imputed']
train['Age'] = train['Age'].replace(0, np.nan)
train['Age'] = train['Age'].fillna(train['Age'].median())

# Landsize imputación
train['Landsize'] = train['Landsize'].replace(0, np.nan)
lands_medians = train.groupby('Rooms_Bathroom')['Landsize'].median()
train['Landsize'] = train.apply(
    lambda row: lands_medians.get(row['Rooms_Bathroom'], row['Landsize']) if pd.isna(row['Landsize']) else row['Landsize'],
    axis=1
)

# sqrt_Distance
train['sqrt_Distance'] = np.sqrt(train['Distance'])

# =========================
# 5. ENTRENAR MODELO LASSO(alpha=100)
# =========================
features = ['Rooms', 'Bathroom', 'Car', 'Landsize', 'BuildingArea_imputed', 'Age', 'sqrt_Distance']
X_train = train[features].fillna(0)
y_train = train["Price"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

lasso = Lasso(alpha=100)
lasso.fit(X_train_scaled, y_train)

# =========================
# 6. PREPROCESAR TEST SET
# =========================
test = test_set.copy()
test['Rooms_Bathroom'] = test['Rooms'].astype(str) + 'R_' + test['Bathroom'].astype(str) + 'B'

# BuildingArea
test['BuildingArea_imputed'] = test.apply(
    lambda row: building_medians_rb.get(row['Rooms_Bathroom'], np.nan) if pd.isna(row['BuildingArea']) else row['BuildingArea'],
    axis=1
)
test['BuildingArea_imputed'] = test.apply(
    lambda row: room_medians.get(row['Rooms'], np.nan) if pd.isna(row['BuildingArea_imputed']) else row['BuildingArea_imputed'],
    axis=1
)

# YearBuilt → Age
test['YearBuilt_imputed'] = test.apply(
    lambda row: year_medians_st.get((row['Suburb'], row['Type']), np.nan) if pd.isna(row['YearBuilt']) else row['YearBuilt'],
    axis=1
)
test['YearBuilt_imputed'] = test.apply(
    lambda row: type_medians.get(row['Type'], np.nan) if pd.isna(row['YearBuilt_imputed']) else row['YearBuilt_imputed'],
    axis=1
)
test['Age'] = 2024 - test['YearBuilt_imputed']
test['Age'] = test['Age'].replace(0, np.nan)
test['Age'] = test['Age'].fillna(train['Age'].median())

# Landsize
test['Landsize'] = test['Landsize'].replace(0, np.nan)
test['Landsize'] = test.apply(
    lambda row: lands_medians.get(row['Rooms_Bathroom'], row['Landsize']) if pd.isna(row['Landsize']) else row['Landsize'],
    axis=1
)

# sqrt_Distance
test['sqrt_Distance'] = np.sqrt(test['Distance'])

X_test = test[features].fillna(0)
X_test_scaled = scaler.transform(X_test)

# =========================
# 7. PREDICCIÓN Y EXPORTACIÓN
# =========================
y_pred = lasso.predict(X_test_scaled)

df_output = pd.DataFrame({'Price': y_pred})
df_output = df_output.reset_index()
df_output.columns = ['index', 'Price']
df_output.to_csv('submission.csv', index=False)

