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


import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.model_selection import *


from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.ensemble import *

from xgboost import *

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import *

import warnings
warnings.filterwarnings("ignore")


# !pip install --pre pycaret


df = pd.read_csv("/kaggle/input/basic-datasets/stroke.csv")


df.head()


np.unique(df['stroke']), np.unique(df['gender']), np.unique(df['work_type']), np.unique(df['Residence_type']), np.unique(df['ever_married'])


df.describe()


df.shape


import seaborn as sns
sns.countplot(x='stroke', data=df)
plt.show()


df.columns


df = pd.get_dummies(df,columns=['gender','ever_married','work_type','Residence_type','smoking_status'])
df.head(4)


from sklearn.impute import KNNImputer

imputer = KNNImputer(missing_values=np.nan)
tab = imputer.fit_transform(df)
df_new = pd.DataFrame(tab, columns=df.columns)
df_new.head(10)


X = df_new.drop('stroke',axis=1)
X.shape


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = df["stroke"].values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
X_test.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Créer un classifieur de forêt aléatoire
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


rf_classifier.fit(X_train, y_train)


y_pred = rf_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.4f}")


print("Rapport de classification :")
print(classification_report(y_test, y_pred))


# import pandas as pd


from pycaret.regression import *


train = pd.read_csv("/kaggle/input/basic-datasets/house_prices.csv")


train.head()


train.columns


train.dtypes


object_columns = train.select_dtypes(include=['object']).columns
object_columns[: 5]


df = pd.get_dummies(train, columns=object_columns, drop_first=True)


df.head(5)


df["SalePrice"][:3]


df = df.drop(columns=["Id"])


imputer = KNNImputer(missing_values=np.nan)
tab = imputer.fit_transform(df)
df_new = pd.DataFrame(tab, columns=df.columns)
df_new.head(5)


X = df_new.drop('SalePrice',axis=1)
X.shape


scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = df_new["SalePrice"].values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
X_test.shape


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Créer un classifieur de forêt aléatoire
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)


rf_regressor.fit(X_train, y_train)


y_pred = rf_regressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"MSE:  {mse:.4f}")


# reg_setup = setup(data=train, target='SalePrice')


# # Comparer tous les modeles
# best_model = compare_models()


# # Créer un modèle de forêt aléatoire
# rf = create_model('rf')

# # Ajuster les hyperparamètres du modèle de forêt aléatoire
# tuned_rf = tune_model(rf, n_jobs=1)


df = pd.read_csv("/kaggle/input/images-in-csv-datasets/hmnist_28_28_RGB.csv")


len(df)


df.head()


plt.imshow(np.array(df.drop(['label'],axis=1)).reshape(len(df), 28, 28, 3)[0])


# X = np.array(df.drop(['label'], axis=1)).reshape(len(df), 28, 28, 3)
# X.shape


# Séparer les caractéristiques et les étiquettes
X = df.drop(columns=['label']).values
y = df['label'].values

# Aplatir les données d'image en vecteurs unidimensionnels
X = X.reshape(len(df), -1)
X.shape, y.shape


# 7 types
np.unique(y)


# # Diviser les données en ensembles d'entraînement et de test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialiser le classifieur XGBoost
# xgb_model = XGBClassifier(
#     n_estimators=100,  
#     learning_rate=0.1, 
#     max_depth=6,  
#     random_state=42,  
#     use_label_encoder=False, 
#     eval_metric='mlogloss' 
# )

# # Entraîner le modèle
# xgb_model.fit(X_train, y_train)

# # Prédire les étiquettes pour l'ensemble de test
# y_pred = xgb_model.predict(X_test)

# # Calculer la précision
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Précision: {accuracy:.4f}")

# # Afficher le rapport de classification
# print("Rapport de classification :")
# print(classification_report(y_test, y_pred))


from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.utils import to_categorical


# Séparer les caractéristiques et les étiquettes
X = df.drop(columns=['label']).values
y = df['label'].values
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)
print(y[: 3])


# Aplatir les données d'image en vecteurs unidimensionnels
X = X.reshape(len(df), -1)  # Transformer (n_samples, 28, 28, 3) en (n_samples, 2352)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Construire le modèle
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),  
    Dense(64, activation='relu'),  
    Dense(num_classes, activation='softmax') 
])

# Compiler le modèle
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.compile(
#     optimizer=Adam(learning_rate=0.001),  
#     loss='categorical_crossentropy',  
#     metrics=[Accuracy()] 
# )

# Afficher la structure du modèle
model.summary()


# Entraîner le modèle
history = model.fit(
    X_train, y_train,
    epochs=20,  
    batch_size=32,  
    validation_split=0.2, 
    verbose=1  # Afficher les informations d'entraînement
)

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Précision sur l'ensemble de test: {accuracy:.4f}")

# Enregistrer le modèle
model.save("/kaggle/working/dense_linear_model_hmnist_rgb.h5")


import tensorflow as tf
from tensorflow import keras


model = Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(num_classes, activation=tf.nn.softmax)
])



# Entraîner le modèle
history = model.fit(
    X_train, y_train,
    epochs=20,  
    batch_size=32,  
    validation_split=0.2, 
    verbose=1  # Afficher les informations d'entraînement
)
model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Précision sur l'ensemble de test: {accuracy:.4f}")

# Enregistrer le modèle
model.save("/kaggle/working/cnn_model_hmnist_rgb.h5")





# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from IPython.core.display import HTML # permet d'afficher du code html dans jupyter
import tensorflow as tf
from tensorflow import keras


df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')


df.head()


columns1 = [f'pixel{i}' for i in range(1, 785)]
X = df[columns1].values  
y = df['label']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


X.shape, y.shape


model = keras.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20)


test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy: ', test_acc*100, "%")

