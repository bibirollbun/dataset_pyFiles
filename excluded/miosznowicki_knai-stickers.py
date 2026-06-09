# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from datetime import datetime


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings('ignore')


path = '/kaggle/input/playground-series-s5e1/'
df_train = pd.read_csv(f'{path}train.csv')
df_test = pd.read_csv(f'{path}test.csv')


#uzupełnienie NaN
# Wypełnianie brakujących wartości medianą dla kombinacji 'country', 'store', 'product'
df_train['num_sold'] = df_train['num_sold'].fillna(df_train.groupby(['country', 'store', 'product'])['num_sold'].transform('median'))



# Grupowanie i walidacja grup z samymi NaN
groups_with_nan_only = (
    df_train.groupby(['country', 'store', 'product'])['num_sold']
    .apply(lambda x: x.isna().all())
    .reset_index(name='all_nan')
    .query('all_nan == True')
)

# Wyświetlanie grup z samymi NaN
print(groups_with_nan_only)


#wypełnienie zerami
df_train['num_sold'].fillna(0, inplace=True)


#y_train kolumna z wynikiem 
y_train = df_train['num_sold']

#usunięcie wyniku z df_train
df_train.drop('num_sold', axis=1, inplace=True)


#teraz df_train i df_test powinny być identyczne

df_train.info()
df_test.info()


#scalanie obu df
df_all = pd.concat([df_train, df_test], axis=0)


def extract_date_features(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['Weekday'] = df['date'].dt.weekday  # 0 = poniedziałek, 6 = niedziela
    df['Is_Weekend'] = df['date'].dt.weekday.isin([5, 6])  # Sobota lub niedziela
    df['Quarter'] = df['date'].dt.quarter
    df['Day_Of_Year'] = df['date'].dt.dayofyear
    df['Week_Of_Year'] = df['date'].dt.isocalendar().week
    df['Is_Month_Start'] = df['date'].dt.is_month_start
    df['Is_Month_End'] = df['date'].dt.is_month_end
    df.drop(column_name, inplace=True, axis = 1)
    return df



df_all = extract_date_features(df_all, 'date')


# Tworzenie sale_id
# identyfikator sprzedaży konkretnej naklejki w konktertnym sklepie i kraju 
df_all['sale_id'] = pd.factorize(df_all['country'] + '_' + df_all['store'] + '_' + df_all['product'])[0]


# usuń krjaj sklep i produkt
df_all.drop('country', axis=1, inplace=True)
df_all.drop('store', axis=1, inplace=True)
df_all.drop('product', axis=1, inplace=True)
df_all.drop('id', axis=1, inplace=True)


# Sprawdzanie unikalnych wartości w każdej kolumnie
unique_values = {col: df_all[col].unique() for col in df_all.columns}

# Wyświetlenie wyników
for column, values in unique_values.items():
    print(f"Kolumna '{column}': {len(values)} unikalnych wartości")
    print(f"Przykładowe wartości: {values[:10]}\n")

#powinno być 90 sale_id


#df dla trenigu i testowy wycięty z df_all o długości df_train

df_train = df_all[:df_train.shape[0]]

# i reszta do trenowania
df_test = df_all[df_train.shape[0]:]


df_train['num_sold'] = y_train


#sprawdzenie
df_train.info()
df_test.info()


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Funkcja do mean target encoding z obsługą braków
def mean_target_encoding(train_df, test_df, target, column):
    """
    Funkcja do kodowania mean target dla kolumny 'column'
    """
    # Oblicz średnią sprzedaż dla każdego unikalnego `sale_id`
    mean_target_map = train_df.groupby(column)[target].mean()
    
    # Kodowanie na podstawie średniej w zbiorze treningowym
    train_encoded = train_df[column].map(mean_target_map)
    
    # Kodowanie w zbiorze testowym na podstawie danych treningowych
    test_encoded = test_df[column].map(mean_target_map)
    
    # Wypełnienie brakujących wartości globalną średnią
    global_mean = train_df[target].mean()
    test_encoded.fillna(global_mean, inplace=True)
    
    return train_encoded, test_encoded

# Walidacja danych
if 'num_sold' not in df_train.columns:
    raise ValueError("Kolumna 'num_sold' nie istnieje w df_train. Upewnij się, że dane są poprawne.")

if 'sale_id' not in df_train.columns or 'sale_id' not in df_test.columns:
    raise ValueError("Kolumna 'sale_id' nie istnieje w danych. Upewnij się, że dane są poprawne.")

# Kodowanie mean target
df_train['sale_id_encoded'], df_test['sale_id_encoded'] = mean_target_encoding(
    train_df=df_train.copy(), 
    test_df=df_test.copy(), 
    target=y_train, 
    column='sale_id'
)

# Usunięcie oryginalnej kolumny `sale_id`
df_train.drop('sale_id', axis=1, inplace=True)
df_test.drop('sale_id', axis=1, inplace=True)

# Dodanie zakodowanej kolumny do zbiorów treningowych
X_train = df_train.join(df_train['sale_id_encoded'])
X_test = df_test.join(df_test['sale_id_encoded'])

# 3. Trening modelu
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 4. Prognozowanie
y_pred = model.predict(X_test)

# Wyświetlenie prognoz
df_test['predicted_num_sold'] = y_pred
print(df_test[['predicted_num_sold']])



y_pred_MTE.shape


from sklearn.model_selection import KFold
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

#X_train df dla trenigu wycięty z df_all o długości df_train

X_train = df_all[:df_train.shape[0]]

# i reszta do trenowania
X_test = df_all[df_train.shape[0]:]


#model = LinearRegression()
model = LGBMRegressor()
model.fit(X_train, y_train)


# Prepare arrays to store out-of-fold predictions and test set predictions
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = LGBMRegressor()
    model.fit(X_tr, y_tr)
    
    # Predict on validation set and test set
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Final averaged predictions for the test set
y_pred = test_preds


# Prepare arrays to store out-of-fold predictions and test set predictions
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = LGBMRegressor()
    model.fit(X_tr, y_tr)
    
    # Predict on validation set and test set
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Final averaged predictions for the test set
y_pred_kfold = test_preds


#trenowanie wrzucone na kaggle

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Split the training data for validation
X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

# Convert to LightGBM dataset format
train_data = lgb.Dataset(X_train_split, label=y_train_split, categorical_feature='auto')
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature='auto', reference=train_data)

# LightGBM parameters
params = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
}

# Define callbacks
callbacks = [
    lgb.early_stopping(stopping_rounds=50),  # Stops training if no improvement
    lgb.log_evaluation(period=10)           # Logs evaluation results every 10 iterations
]

# Train model
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=1000,
    callbacks=callbacks  # No need for verbose_eval as log_evaluation handles it
)

# Save the model
model.save_model('lightgbm_regressor.txt')

# Predict on test data
y_pred = model.predict(X_test, num_iteration=model.best_iteration)


y_pred = model.predict(X_test)


y_pred.min()


y_pred_MTE.shape


y_pred_kfold


ssub = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


ssub['num_sold'] = y_pred_kfold


ssub.to_csv('submission.csv', index = False)


ssub.shape




