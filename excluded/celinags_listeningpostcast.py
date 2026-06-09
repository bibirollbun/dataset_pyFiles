# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
import psutil
import seaborn as sns
import os
import xgboost as xgb
import category_encoders as ce
import xgboost as xgb
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Cargamos el dataset de entrenamiento y test
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


# Copias de trabajo
train_df = train.copy()
test_df = test.copy()

print(train_df.shape)
print(train_df.columns)
train_df.head()
train_df.info()
train_df.isnull().sum()


# Limpieza inicial
# Detectamos filas duplicadas
duplicadas = train_df.duplicated()
print(f"Filas duplicadas en el dataset de entrenamiento: {duplicadas.sum()}")

train_df[duplicadas]

duplicado = test_df.duplicated()
print(f"Filas duplicadas en el dataset de validaciÃ³n: {duplicado.sum()}")

# Rellenamos valores faltantes bÃ¡sicos
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median())
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median())
train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median())

#Hacemos lo mismo para el test
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median())
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median())
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median())




# Transformamos `Publication_Time` en hora numÃ©rica
# Revisamos si hay valores no convertibles como "Night"

print("Valores Ãºnicos en Publication_Time:", train_df['Publication_Time'].unique())

# Definimos el mapeo textual â†’ hora numÃ©rica
time_mapping = {
    'Morning':   9,   # 09:00
    'Afternoon': 15,  # 15:00
    'Evening':   19,  # 19:00
    'Night':     22   # 22:00
}

# Aplicamos el mapeo directamente
train_df['Publication_Hour'] = train_df['Publication_Time'].map(time_mapping)
test_df ['Publication_Hour'] = test_df ['Publication_Time'].map(time_mapping)

# (Eliminamos Publication_Time si no vamos a usarlo mÃ¡s directamente)
train_df = train_df.drop(columns=['Publication_Time'])
test_df = test_df.drop(columns=['Publication_Time'])

# Creamos una nueva variable Publication_Day_Num y luego aplicamos el mapeo
day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

train_df['Publication_Day_Num'] = train_df['Publication_Day'].map(day_mapping)
test_df['Publication_Day_Num'] = test_df['Publication_Day'].map(day_mapping)

# (Eliminamos Publication_Day si no vamos a usarlo mÃ¡s directamente)
train_df = train_df.drop(columns=['Publication_Day'])
test_df = test_df.drop(columns=['Publication_Day'])





# Para unir los datasets y calcular estadÃ­sticas comunes
train_df['source'] = 'train'
test_df['source'] = 'test'
test_df['Listening_Time_minutes'] = np.nan  # Para poder hacer agregados

combined = pd.concat([train_df, test_df])

# Agrupamos por Podcast_Name y sacamos estadÃ­sticas Ãºtiles
podcast_stats = combined.groupby('Podcast_Name').agg({
    'Listening_Time_minutes': 'mean',
    'Host_Popularity_percentage': 'mean',
    'id': 'count'
}).rename(columns={
    'Listening_Time_minutes': 'mean_listening_time_by_podcast',
    'Host_Popularity_percentage': 'mean_host_popularity_by_podcast',
    'id': 'podcast_episode_count'
}).reset_index()

# Unimos estas nuevas features al dataset original
combined = combined.merge(podcast_stats, on='Podcast_Name', how='left')

# Volvemos a separar en train y test
train_df = combined[combined['source'] == 'train'].drop(columns=['source'])
test_df = combined[combined['source'] == 'test'].drop(columns=['source', 'Listening_Time_minutes'])



# CodificaciÃ³n por promedio de la variable target para Podcast_Name
podcast_target_encoding = train_df.groupby('Podcast_Name')['Listening_Time_minutes'].mean().to_dict()
train_df['Podcast_Target_Encoded'] = train_df['Podcast_Name'].map(podcast_target_encoding)
test_df['Podcast_Target_Encoded'] = test_df['Podcast_Name'].map(podcast_target_encoding)

# Si alguna categorÃ­a en test no estÃ¡ en train, rellenamos con la media global
media_global = train_df['Listening_Time_minutes'].mean()
test_df['Podcast_Target_Encoded'] = test_df['Podcast_Target_Encoded'].fillna(media_global)

# CodificaciÃ³n por promedio (target encoding) para Genre
genre_encoder = TargetEncoder()
train_df['Genre_Encoded'] = genre_encoder.fit_transform(train_df['Genre'], train_df['Listening_Time_minutes'])
test_df['Genre_Encoded'] = genre_encoder.transform(test_df['Genre'])

# AsegurÃ©monos de que las categorÃ­as desconocidas se codifiquen con la media global, si es necesario
# Si quieres verificar que la codificaciÃ³n se realizÃ³ correctamente en el conjunto de prueba
print(test_df['Podcast_Target_Encoded'].isnull().sum())  # DeberÃ­a ser 0
# Verificamos si el conjunto de prueba tiene alguna categorÃ­a no vista durante el entrenamiento
unknown_podcast_names = test_df[~test_df['Podcast_Name'].isin(train_df['Podcast_Name'])]
print("CategorÃ­as no vistas en el conjunto de prueba:")
print(unknown_podcast_names['Podcast_Name'].unique())

# Eliminamos columnas originales si ya no se usan
train_df.drop(columns=['Podcast_Name', 'Genre'], inplace=True)
test_df.drop(columns=['Podcast_Name', 'Genre'], inplace=True)



# Verificamos si hay valores faltantes en la columna 'Podcast_Target_Encoded'
print(train_df['Podcast_Target_Encoded'].isnull().sum())  # DeberÃ­a ser 0
print(test_df['Podcast_Target_Encoded'].isnull().sum())   # DeberÃ­a ser 0 (si tienes valores no presentes en train, se rellenarÃ¡n con la media global)



# Verificamos si hay valores faltantes en la columna 'Podcast_Target_Encoded'
print(train_df['Podcast_Target_Encoded'].isnull().sum())  # DeberÃ­a ser 0
print(test_df['Podcast_Target_Encoded'].isnull().sum())   # DeberÃ­a ser 0 (si tienes valores no presentes en train, se rellenarÃ¡n con la media global)



# Verificamos que la columna 'Publication_Hour' no tenga valores nulos ni incorrectos
print(train_df['Publication_Hour'].isnull().sum())  # DeberÃ­a ser 0
print(train_df['Publication_Hour'].unique())       # Verifica que todos los valores estÃ©n bien mapeados: [9, 15, 19, 22]



# Verificamos que la columna 'Publication_Day_Num' no tenga valores nulos
print(train_df['Publication_Day_Num'].isnull().sum())  # DeberÃ­a ser 0
print(train_df['Publication_Day_Num'].unique())        # DeberÃ­a ser valores entre 0 y 6 (lunes = 0, domingo = 6)


# Verificamos las primeras filas del dataframe para asegurarnos de que las transformaciones fueron aplicadas correctamente
print(train_df.head())

# Comprobamos si hay valores nulos en todo el dataframe (deberÃ­an ser cero en todas las columnas)
print(train_df.isnull().sum())

# Verificamos los tipos de las columnas para asegurarnos de que las transformaciones han sido correctas
print(train_df.dtypes)




from sklearn.preprocessing import LabelEncoder
# SelecciÃ³n de variables
X = train_df.drop(columns=['id', 'Episode_Title', 'Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']



from sklearn.preprocessing import LabelEncoder

# Codificar variables categÃ³ricas
encoder = LabelEncoder()
encoder_sentiment = LabelEncoder()

from sklearn.preprocessing import LabelEncoder

# Para Episode_Sentiment
encoder_sentiment = LabelEncoder()
X['Episode_Sentiment'] = encoder_sentiment.fit_transform(X['Episode_Sentiment'])

# Convertimos columnas booleanas a enteros (0 y 1)
bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)





# ---------- NUEVAS FEATURES ----------
# InteracciÃ³n entre popularidad del host y del invitado

encoder = LabelEncoder()

X['HostGuest_Popularity_interaction'] = X['Host_Popularity_percentage'] * X['Guest_Popularity_percentage']

# Densidad de anuncios: cantidad de anuncios por minuto del episodio
X['Ads_per_minute'] = X['Number_of_Ads'] / (X['Episode_Length_minutes'] + 1e-5)  # evitar divisiÃ³n por cero

# Horario de publicaciÃ³n categorizado
X['Publication_Hour_Group'] = pd.cut(
    X['Publication_Hour'], bins=[-1, 6, 13, 18, 24],
    labels=["Madrugada", "MaÃ±ana", "Tarde", "Noche"]
)
encoder_hour = LabelEncoder()
X['Publication_Hour_Group'] = encoder_hour.fit_transform(X['Publication_Hour_Group'])

# =================== ENTRENAR MODELO BÃ�SICO PARA IMPORTANCIAS ===================
modelo_basico = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=42,
    n_estimators=200,
    max_depth=6
)
modelo_basico.fit(X, y)

# =================== TOP 10 FEATURES ===================
importances = modelo_basico.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]
top_features = features[indices[:10]]

print("\nTop 10 Features:")
print(top_features)

# =================== ESCALAR SOLO TOP FEATURES ===================
X_top = X[top_features]
scaler = StandardScaler()
X_top_scaled = scaler.fit_transform(X_top)

# =================== DIVIDIR DATOS ===================
X_train, X_valid, y_train, y_valid = train_test_split(X_top_scaled, y, test_size=0.2, random_state=42)
# Para evaluaciones finales
X_train_gs, X_valid_gs, y_train_gs, y_valid_gs = X_train, X_valid, y_train, y_valid
# =================== RANDOMIZEDSEARCH XGBOOST ===================
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'reg_lambda': [0, 0.1, 1],
    'reg_alpha': [0, 0.1, 1]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,  # Aumentado para mejor bÃºsqueda
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# =================== EVALUAR XGBOOST OPTIMIZADO ===================
xgb_preds = best_model.predict(X_valid)
xgb_rmse = mean_squared_error(y_valid, xgb_preds, squared=False)
print(f"\nRMSE con XGBoost optimizado: {xgb_rmse:.4f}")
print("Mejores parÃ¡metros:", random_search.best_params_)

# =================== LIGHTGBM ===================
lgb_model = LGBMRegressor(
    objective='regression',
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6
)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_valid)
lgb_rmse = mean_squared_error(y_valid, lgb_preds, squared=False)
print(f"RMSE con LightGBM: {lgb_rmse:.4f}")

# =================== STACKING REGRESSOR ===================
base_models = [
    ('xgb', best_model),
    ('lgb', lgb_model)
]
meta_model = LinearRegression()

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    n_jobs=-1
)
stacking_model.fit(X_train, y_train)
stacking_preds = stacking_model.predict(X_valid)
stacking_rmse = mean_squared_error(y_valid, stacking_preds, squared=False)
print(f"RMSE del modelo de Stacking: {stacking_rmse:.4f}")

# =================== VOTING REGRESSOR ===================
voting_model = VotingRegressor(
    estimators=[
        ('xgb', best_model),
        ('lgb', lgb_model)
    ],
    n_jobs=-1
)
voting_model.fit(X_train, y_train)
voting_preds = voting_model.predict(X_valid)

voting_rmse = mean_squared_error(y_valid, voting_preds, squared=False)
voting_mae = mean_absolute_error(y_valid, voting_preds)
voting_r2 = r2_score(y_valid, voting_preds)

print("\n[ Voting Regressor ]")
print(f"RMSE: {voting_rmse:.4f}")
print(f"MAE: {voting_mae:.4f}")
print(f"RÂ² Score: {voting_r2:.4f}")

# =================== CROSS-VALIDATION ===================
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
voting_cv_scores = cross_val_score(voting_model, X_top_scaled, y, cv=kfold, scoring='neg_root_mean_squared_error')
print(f"Cross-Validation RMSE promedio (Voting): {-voting_cv_scores.mean():.4f}")

# =================== IMPORTANCIA DE CARACTERÃ�STICAS FINAL ===================
xgb.plot_importance(best_model, max_num_features=10, importance_type='weight', height=0.8)
plt.title("Importancia de caracterÃ­sticas (Top 10)")
plt.show()




# =================== PROCESAMIENTO DEL CONJUNTO DE TEST ===================

# Copiar el test y eliminar columnas innecesarias
X_test = test_df.drop(columns=['id', 'Episode_Title']).copy()

# Agrupar horas exactamente igual al entrenamiento
X_test['Publication_Hour_Group'] = pd.cut(
    X_test['Publication_Hour'], bins=[-1, 6, 13, 18, 24],
    labels=["Madrugada", "MaÃ±ana", "Tarde", "Noche"]
)

# Aplicar los mismos LabelEncoders usados en entrenamiento
X_test['Episode_Sentiment'] = encoder_sentiment.transform(X_test['Episode_Sentiment'])
X_test['Publication_Hour_Group'] = encoder_hour.transform(X_test['Publication_Hour_Group'])

# Asegurar el mismo tipo para las columnas booleanas
X_test[bool_cols] = X_test[bool_cols].astype(int)

# Crear variables derivadas igual que en entrenamiento
X_test['HostGuest_Popularity_interaction'] = (
    X_test['Host_Popularity_percentage'] * X_test['Guest_Popularity_percentage']
)
X_test['Ads_per_minute'] = X_test['Number_of_Ads'] / (X_test['Episode_Length_minutes'] + 1e-5)

# Seleccionar las 10 features importantes
X_test_top = X_test[top_features].copy()

# Escalar con el mismo scaler del entrenamiento
X_test_scaled = scaler.transform(X_test_top)

# =================== PREDICCIONES FINALES CON XGBOOST OPTIMIZADO ===================
# Realizar predicciones con el mejor modelo encontrado
predicciones_finales = best_model.predict(X_test_scaled)

# Clipping: asegurar que estÃ©n en el rango [0, 120]
predicciones_finales = np.clip(predicciones_finales, 0, 120)

# Agregar predicciones al dataframe para generar submission
submission = test_df[['id']].copy()
submission['Listening_Time_minutes'] = predicciones_finales

# Revisar rango de predicciones
min_pred = predicciones_finales.min()
max_pred = predicciones_finales.max()

print(f"ğŸ”� PredicciÃ³n mÃ­nima: {min_pred:.2f}")
print(f"ğŸ”� PredicciÃ³n mÃ¡xima: {max_pred:.2f}")

if min_pred < 0 or max_pred > 120:
    print("âš ï¸� AtenciÃ³n: Hay predicciones fuera del rango esperado (0-120 minutos).")
else:
    print("âœ… Todas las predicciones estÃ¡n dentro del rango razonable.")

# Ajustar predicciones fuera de rango: mÃ­nimo 0, mÃ¡ximo 120
predicciones_finales_clip = np.clip(predicciones_finales, 0, 120)

# Crear archivo de submission para Kaggle
submission = test_df[['id']].copy()
submission['Listening_Time_minutes'] = predicciones_finales

# Exportar a CSV
submission.to_csv('submission_xgboost.csv', index=False)

print("ğŸ“� Archivo 'submission_xgboost.csv' generado correctamente.")


