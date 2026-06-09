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

train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train


train.describe()


train.dtypes


train.isnull().sum()


# # Remplir les valeurs manquantes de 'Episode_Length_minutes' avec la médiane
# train['Episode_Length_minutes'] = train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median())

# # Remplir les valeurs manquantes de 'Guest_Popularity_percentage' avec la médiane
# train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median())

# # Remplir les valeurs manquantes de 'Number_of_Ads' avec la médiane
# train['Number_of_Ads'] = train['Number_of_Ads'].fillna(train['Number_of_Ads'].median())

# # Vérifier les valeurs manquantes après imputation
# print(train.isnull().sum())


import seaborn as sns
import matplotlib.pyplot as plt

# Variables numériques
train.select_dtypes(include=['float64', 'int64']).hist(bins=15, figsize=(15, 10))
plt.show()

# Variables catégorielles
categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train, x=col)
    plt.title(f'Distribution de {col}')
    plt.xticks(rotation=45)
    plt.show()


#Encodage des variables catégorielles

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = label_encoder.fit_transform(train[col])
    test[col] = label_encoder.transform(test[col])


from sklearn.preprocessing import MinMaxScaler

# Scaler for the target variable
target_scaler = MinMaxScaler()

# Normalize the target variable (Listening_Time_minutes) in the train dataset
train['Listening_Time_minutes'] = target_scaler.fit_transform(train[['Listening_Time_minutes']])

scaler = MinMaxScaler()

# Sélectionner les colonnes numériques dans train
numerical_cols_train = train.select_dtypes(include=['float64', 'int64']).columns
numerical_cols_train = numerical_cols_train.drop(['Listening_Time_minutes', 'id'], errors='ignore')

# Vérifier les colonnes communes entre train et test
numerical_cols_test = test.select_dtypes(include=['float64', 'int64']).columns
common_cols = numerical_cols_train.intersection(numerical_cols_test)


# Appliquer la normalisation uniquement sur les colonnes communes
train[common_cols] = scaler.fit_transform(train[common_cols])
test[common_cols] = scaler.transform(test[common_cols])


train


X_train = train.drop(columns=['id', 'Listening_Time_minutes'])
y_train = train['Listening_Time_minutes']
X_test = test.drop(columns=['id'])


from sklearn.model_selection import train_test_split

# Séparer le jeu d'entraînement en données d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import numpy as np

# XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)
xgb_rmse = mean_squared_error(y_val, xgb_pred, squared=False)
print(f'XGBoost RMSE: {xgb_rmse:.4f}')

# LightGBM
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_val)
lgb_rmse = mean_squared_error(y_val, lgb_pred, squared=False)
print(f'LightGBM RMSE: {lgb_rmse:.4f}')

# CatBoost
cat_model = CatBoostRegressor(verbose=0, random_state=42)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_val)
cat_rmse = mean_squared_error(y_val, cat_pred, squared=False)
print(f'CatBoost RMSE: {cat_rmse:.4f}')

# Sélection du meilleur modèle
rmses = {'XGBoost': xgb_rmse, 'LightGBM': lgb_rmse, 'CatBoost': cat_rmse}
best_model_name = min(rmses, key=rmses.get)
print(f'\n✅ Meilleur modèle : {best_model_name} avec RMSE = {rmses[best_model_name]:.4f}')

# Utiliser le meilleur modèle pour les prédictions test
best_model = {'XGBoost': xgb_model, 'LightGBM': lgb_model, 'CatBoost': cat_model}[best_model_name]


y_test_pred = best_model.predict(X_test)


# Création du fichier de soumission

y_test_pred_reshaped = y_test_pred.reshape(-1, 1)
y_test_inverse_transform = target_scaler.inverse_transform(y_test_pred_reshaped)

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': y_test_inverse_transform.flatten()
})


submission.to_csv('submission.csv', index=False)



submission.head(10)




