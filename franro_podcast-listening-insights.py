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



#cargamos los datos
df_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


print(df_train.info())


df_train.describe()


#informaci贸n variables categ贸ricas
colum_categor = ['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment']
for col in colum_categor:
    print(f'\nEstad铆stica de {col}:')
    print(f"Top 10 valores m谩s frecuentes:")
    value_counts = df_train[col].value_counts().head(10)  # Top 10 valores m谩s comunes
    print(value_counts)
    print(f"N煤mero total de valores 煤nicos: {df_train[col].nunique()}")


# Valores faltantes
missing_values = df_train.isnull().sum()
print("\nValores faltantes por columna:")
print(missing_values)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Configuraci贸n de visualizaci贸n
plt.figure(figsize=(15, 10))

# Variables num茅ricas - histogramas
numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']

for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(df_train[col].dropna(), kde=True)
    plt.title(f'Distribuci贸n de {col}')
    plt.tight_layout()

plt.figure(figsize=(15, 10))
# Variables categ贸ricas - countplots
for i, col in enumerate(['Genre', 'Publication_Day', 'Episode_Sentiment']):
    plt.subplot(2, 2, i+1)
    top_categories = df_train[col].value_counts().head(10).index
    sns.countplot(y=df_train[df_train[col].isin(top_categories)][col])
    plt.title(f'Top 10 m谩s frecuentes: {col}')
    plt.tight_layout()

plt.show()






# Matriz de correlaci贸n para variables num茅ricas
plt.figure(figsize=(10, 8))
correlation_matrix = df_train[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlaci贸n')

# Relaci贸n entre variables categ贸ricas y Listening_Time_minutes
plt.figure(figsize=(15, 12))
for i, col in enumerate(['Genre', 'Publication_Day', 'Episode_Sentiment']):
    plt.subplot(2, 2, i+1)
    top_categories = df_train[col].value_counts().head(10).index
    sns.boxplot(x=df_train[df_train[col].isin(top_categories)][col], 
                y=df_train[df_train[col].isin(top_categories)]['Listening_Time_minutes'])
    plt.title(f'{col} vs Listening Time')
    plt.xticks(rotation=45)
    plt.tight_layout()

# Scatter plots para relaciones entre variables num茅ricas
plt.figure(figsize=(15, 12))
for i, x_col in enumerate(['Episode_Length_minutes', 'Host_Popularity_percentage', 
                          'Guest_Popularity_percentage', 'Number_of_Ads']):
    plt.subplot(2, 2, i+1)
    sns.scatterplot(data=df_train.sample(5000), x=x_col, y='Listening_Time_minutes', alpha=0.5)
    plt.title(f'{x_col} vs Listening Time')
    plt.tight_layout()

plt.show()


# Pair plot para m煤ltiples variables num茅ricas
subset_df = df_train.sample(5000)  # Muestra para reducir tiempo de computaci贸n
sns.pairplot(subset_df[numeric_cols])
plt.suptitle('Relaciones entre Variables Num茅ricas', y=1.02)

# An谩lisis por grupos
plt.figure(figsize=(15, 10))
for i, cat_col in enumerate(['Genre', 'Publication_Day']):
    top_cats = df_train[cat_col].value_counts().head(5).index
    data = df_train[df_train[cat_col].isin(top_cats)]
    
    plt.subplot(1, 2, i+1)
    for cat in top_cats:
        subset = data[data[cat_col] == cat]
        sns.kdeplot(subset['Listening_Time_minutes'], label=cat)
    
    plt.title(f'Distribuci贸n de Listening Time por {cat_col}')
    plt.legend()
    plt.tight_layout()

plt.show()


# 1. Visualizar los valores faltantes para entender su distribuci贸n

plt.figure(figsize=(10, 6))
sns.heatmap(df_train.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Mapa de valores faltantes')
plt.tight_layout()
plt.show()


# Para Number_of_Ads (solo 1 valor faltante)
df_train['Number_of_Ads'].fillna(df_train['Number_of_Ads'].median(), inplace=True)


# Imputaci贸n para Episode_Length_minutes por g茅nero
# Paso 1: Calcular las medianas por g茅nero
genre_medians = df_train.groupby('Genre')['Episode_Length_minutes'].median()

# Paso 2: Crear un diccionario de mapeo de g茅nero a mediana
genre_median_dict = genre_medians.to_dict()

# Paso 3: Funci贸n para asignar la mediana correspondiente seg煤n el g茅nero
def fill_episode_length(row):
    if pd.isna(row['Episode_Length_minutes']):
        genre = row['Genre']
        # Si el g茅nero tiene una mediana calculada, usar esa
        if genre in genre_median_dict and not pd.isna(genre_median_dict[genre]):
            return genre_median_dict[genre]
        # Si no, usar la mediana general
        else:
            return df_train['Episode_Length_minutes'].median()
    # Si no es NA, mantener el valor original
    return row['Episode_Length_minutes']

# Paso 4: Aplicar la funci贸n a cada fila
df_train['Episode_Length_minutes'] = df_train.apply(fill_episode_length, axis=1)

# Verificar que no queden valores faltantes
print(f"Valores faltantes despu茅s de imputaci贸n: {df_train['Episode_Length_minutes'].isnull().sum()}")


# Para Guest_Popularity_percentage y Number_of_Ads tambi茅n usaremos la mediana
guest_popularity_median = df_train['Guest_Popularity_percentage'].median()
print(f"Mediana de Guest_Popularity_percentage: {guest_popularity_median:.2f}%")
df_train['Guest_Popularity_percentage'].fillna(guest_popularity_median, inplace=True)


# Verificar que no queden valores faltantes
print("\nValores faltantes despu茅s de imputaci贸n:")
print(df_train.isnull().sum())


# convertir las variables categ贸ricas en num茅rica
# Codificaci贸n de variables categ贸ricas
df_train_encoded = pd.get_dummies(df_train, columns=['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'], drop_first=True)

#vamos a revisar el dataframe nuevo
df_train_encoded.head()


# Separar variables predictoras (X) y variable objetivo (y)
X = df_train_encoded.drop(columns=['Listening_Time_minutes', 'id'])  # Quitamos la variable objetivo y 'id' que no es relevante
y = df_train_encoded['Listening_Time_minutes']


from sklearn.model_selection import train_test_split

# Divisi贸n en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verifica las dimensiones
print(f'Tama帽o del conjunto de entrenamiento: {X_train.shape}')
print(f'Tama帽o del conjunto de prueba: {X_test.shape}')


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Crear el modelo de regresi贸n lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluaci贸n del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error cuadr谩tico medio (MSE): {mse}')
print(f'R2: {r2}')


from sklearn.ensemble import RandomForestRegressor

# Creamos el modelo
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenamos el modelo
rf_model.fit(X_train, y_train)

# Predecimos con los datos de prueba
y_pred_rf = rf_model.predict(X_test)

# Evaluamos el rendimiento
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("���� Random Forest")
print(f"Error cuadr谩tico medio (MSE): {mse_rf}")
print(f"R2: {r2_rf}")

