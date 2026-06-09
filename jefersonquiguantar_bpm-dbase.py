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

# Cargar los datasets
df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# Muestra las primeras 5 filas del conjunto de entrenamiento
print("Primeras 5 filas del dataset de entrenamiento:")
print(df_train.head())

# Obtén información sobre las columnas y los tipos de datos
print("\nInformación del dataset de entrenamiento:")
print(df_train.info())


import pandas as pd
import numpy as np

# Cargar los datos
df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# Visualizar las primeras filas y obtener información general
print("Información del conjunto de entrenamiento:")
print(df_train.head())
print("\nColumnas y tipos de datos:")
print(df_train.info())

# Verificar si hay valores nulos
print("\nValores nulos en el conjunto de entrenamiento:")
print(df_train.isnull().sum())

# Describir las estadísticas de las columnas numéricas
print("\nEstadísticas descriptivas:")
print(df_train.describe())
print("Columnas en df_train:", df_train.columns)
print("Columnas en df_test:", df_test.columns)


target = 'BeatsPerMinute'
y_train = df_train[target]

# Elige el resto de las columnas como las características (X)
# Excluye 'id' y la variable objetivo
features = [col for col in df_train.columns if col not in ['id', target]]
X_train = df_train[features]

# Para el conjunto de prueba, solo necesitas las características
X_test = df_test[features]

print(f"Número de características (columnas de X): {len(features)}")
print(f"Columnas de características: {features}")

