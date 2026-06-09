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

# Cargar los datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Verifica el contenido
display(train.head())
display(test.head())


print(train.info())


print(train.describe())


# Supongamos que 'target' es tu variable objetivo
correlations = train.corr(numeric_only=True)['Calories'].sort_values(ascending=False)
display(correlations)


display(train.groupby('Sex')['Calories'].mean())


display(train.groupby('Sex')['Calories'].min())
display(train.groupby('Sex')['Calories'].max())


import matplotlib.pyplot as plt

plt.hist(train['Calories'], bins=50, edgecolor='black')
plt.xlabel('Calories')
plt.ylabel('Frecuencia')
plt.title('Histograma de Calories')
plt.show()


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# 1. Separar features y target
X = train.drop(['Calories','id','Height','Weight'], axis=1)
y = train['Calories']


# 2. Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# 3. Identificar columnas categóricas
cat_features = ['Sex']
num_features = [col for col in X.columns if col not in cat_features]



# 4. Crear transformador de columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ],
    remainder='passthrough'  # Deja las columnas numéricas sin tocar
)


# 5. Pipeline con modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# 6. Entrenar
pipeline.fit(X_train, y_train)


# 7. Predecir y evaluar con RMSLE
def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

y_val_pred = pipeline.predict(X_val)
print("RMSLE en validación:", rmsle(y_val, y_val_pred))


# 8. Predecir sobre test y exportar submission
test_pred = pipeline.predict(test)
test_pred = np.maximum(0, test_pred)


sample_submission['Calories'] = test_pred
sample_submission.to_csv('submission.csv', index=False)




