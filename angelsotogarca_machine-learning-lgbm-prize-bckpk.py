import numpy as np
import pandas as pd 
import os
from sklearn.preprocessing import RobustScaler
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
train = pd.concat([train, train_extra], ignore_index=True)


train_categorical_columns = train.select_dtypes(include=['object']).columns.tolist()
test_categorical_columns = test.select_dtypes(include=['object']).columns.tolist()


X = train.drop('Price', axis=1)
y = train['Price']


# 1. Importaciones necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import lightgbm as lgb
import optuna
from IPython.display import FileLink

# 2. Preprocesamiento inicial - Train
X['Weight Capacity (kg)'] = X['Weight Capacity (kg)'].fillna(X['Weight Capacity (kg)'].mean())
X['Compartments'] = X['Compartments'].fillna(X['Compartments'].mean())
X['Brand'] = X['Brand'].fillna(X['Brand'].mode()[0])
X['Material'] = X['Material'].fillna(X['Material'].mode()[0])
X['Size'] = X['Size'].fillna(X['Size'].mode()[0])
X['Laptop Compartment'] = X['Laptop Compartment'].fillna(X['Laptop Compartment'].mode()[0])
X['Waterproof'] = X['Waterproof'].fillna(X['Waterproof'].mode()[0])
X['Style'] = X['Style'].fillna(X['Style'].mode()[0])
X['Color'] = X['Color'].fillna(X['Color'].mode()[0])

# 3. Codificación One-Hot para Train
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_data = X[categorical_columns]
encoded_data = encoder.fit_transform(categorical_data)

encoded_feature_names = []
for i, col in enumerate(categorical_columns):
   categories = encoder.categories_[i]
   for category in categories:
       encoded_feature_names.append(f"{col}_{category}")

encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=X.index)
X_numeric = X.drop(categorical_columns, axis=1)
X_encoded = pd.concat([X_numeric, encoded_df], axis=1)

# 4. Feature Engineering para Train
X_processed = X_encoded.copy()
X_processed['Compartments'] = X_processed['Compartments'].clip(lower=0)
X_processed['Weight Capacity (kg)'] = X_processed['Weight Capacity (kg)'].clip(lower=0)

# Aplicar log1p con manejo seguro de valores
X_processed['Compartments_log'] = np.log1p(X_processed['Compartments'].replace(0, np.finfo(float).eps))
X_processed['Weight_Capacity_log'] = np.log1p(X_processed['Weight Capacity (kg)'].replace(0, np.finfo(float).eps))

# Guardar el scaler para usar en test
scaler = RobustScaler()
X_processed[['Compartments_log', 'Weight_Capacity_log']] = scaler.fit_transform(
   X_processed[['Compartments_log', 'Weight_Capacity_log']]
)

# Características adicionales
X_processed['Compartments_per_kg'] = X_processed['Compartments_log'] / (X_processed['Weight_Capacity_log'] + np.finfo(float).eps)
X_processed['Brand_Premium'] = X_processed[['Brand_Nike', 'Brand_Adidas', 'Brand_Under Armour']].max(axis=1)
X_processed['Material_Synthetic'] = X_processed[['Material_Nylon', 'Material_Polyester']].max(axis=1)
X_processed['Material_Natural'] = X_processed[['Material_Leather', 'Material_Canvas']].max(axis=1)

X_processed = X_processed.fillna(0)

# 5. División de datos
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 6. Optimización con Optuna
study = optuna.create_study(direction='minimize')
def objective(trial):
   params = {
       'objective': 'regression',
       'boosting_type': 'gbdt',
       'max_depth': trial.suggest_int('max_depth', 3, 8),
       'num_leaves': trial.suggest_int('num_leaves', 20, 150),
       'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
       'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05),
       'subsample': trial.suggest_float('subsample', 0.6, 1.0),
       'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
       'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 50.0),
       'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 50.0),
       'random_state': 42
   }
   model = lgb.LGBMRegressor(**params)
   model.fit(X_train, y_train)
   pred = model.predict(X_val)
   return np.sqrt(mean_squared_error(y_val, pred))

study.optimize(objective, n_trials=50)
best_params = study.best_params
best_params['random_state'] = 42

# 7. Entrenamiento con validación cruzada
lgb_model = lgb.LGBMRegressor(**best_params)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_train = X_train.shape[0]
lgb_oof = np.zeros(n_train)

for train_idx, val_idx in kf.split(X_train):
   X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
   X_vl = X_train.iloc[val_idx]
   lgb_model.fit(X_tr, y_tr)
   lgb_oof[val_idx] = lgb_model.predict(X_vl)

# 8. Modelo meta
stacked_train = pd.DataFrame(lgb_oof, columns=['lgb_pred'], index=X_train.index)
meta_model = LinearRegression()
meta_model.fit(stacked_train, y_train)

# 9. Preprocesamiento de test
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean())
test['Compartments'] = test['Compartments'].fillna(test['Compartments'].mean())
test['Brand'] = test['Brand'].fillna(test['Brand'].mode()[0])
test['Material'] = test['Material'].fillna(test['Material'].mode()[0])
test['Size'] = test['Size'].fillna(test['Size'].mode()[0])
test['Laptop Compartment'] = test['Laptop Compartment'].fillna(test['Laptop Compartment'].mode()[0])
test['Waterproof'] = test['Waterproof'].fillna(test['Waterproof'].mode()[0])
test['Style'] = test['Style'].fillna(test['Style'].mode()[0])
test['Color'] = test['Color'].fillna(test['Color'].mode()[0])

# 10. Aplicar el mismo procesamiento al test
categorical_data = test[categorical_columns]
encoded_data = encoder.transform(categorical_data)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=test.index)
X_numeric = test.drop(categorical_columns, axis=1)
X_encoded = pd.concat([X_numeric, encoded_df], axis=1)

# 11. Feature Engineering para test
X_encoded['Compartments'] = X_encoded['Compartments'].clip(lower=0)
X_encoded['Weight Capacity (kg)'] = X_encoded['Weight Capacity (kg)'].clip(lower=0)

X_encoded['Compartments_log'] = np.log1p(X_encoded['Compartments'].replace(0, np.finfo(float).eps))
X_encoded['Weight_Capacity_log'] = np.log1p(X_encoded['Weight Capacity (kg)'].replace(0, np.finfo(float).eps))

X_encoded[['Compartments_log', 'Weight_Capacity_log']] = scaler.transform(
   X_encoded[['Compartments_log', 'Weight_Capacity_log']]
)

X_encoded['Compartments_per_kg'] = X_encoded['Compartments_log'] / (X_encoded['Weight_Capacity_log'] + np.finfo(float).eps)
X_encoded['Brand_Premium'] = X_encoded[['Brand_Nike', 'Brand_Adidas', 'Brand_Under Armour']].max(axis=1)
X_encoded['Material_Synthetic'] = X_encoded[['Material_Nylon', 'Material_Polyester']].max(axis=1)
X_encoded['Material_Natural'] = X_encoded[['Material_Leather', 'Material_Canvas']].max(axis=1)

# Asegurar mismas columnas que en train
missing_cols = set(X_processed.columns) - set(X_encoded.columns)
for col in missing_cols:
   X_encoded[col] = 0

# Ordenar columnas igual que en train
X_encoded = X_encoded[X_processed.columns]
X_encoded = X_encoded.fillna(0)

# 12. Predicciones
lgb_pred = lgb_model.predict(X_encoded)
test_features = pd.DataFrame(lgb_pred, columns=['lgb_pred'], index=X_encoded.index)
y_pred = meta_model.predict(test_features)

# 13. Crear submission
sub = pd.DataFrame()
sub['id'] = test['id']
sub['pred'] = y_pred
sub.to_csv('submission9.csv', index=False)

# 14. Crear enlace de descarga
display(FileLink('submission9.csv'))

