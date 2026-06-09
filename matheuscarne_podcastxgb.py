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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

print("Train dataset: ", df_train.shape)
print("Test dataset:", df_test.shape)
print("Submission dataset:", df_submission.shape)


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Overview
display(df_train.head())
print(df_train.shape)

# 2. Target Variable Analysis
display(df_train['Listening_Time_minutes'].describe())
plt.figure(figsize=(10, 6))
sns.histplot(df_train['Listening_Time_minutes'], kde=True)
plt.title('Distribution of Listening Time')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(y=df_train['Listening_Time_minutes'])
plt.title('Boxplot of Listening Time')
plt.show()


# 3. Missing Values
print(df_train.isnull().sum())
plt.figure(figsize=(12, 6))
sns.heatmap(df_train.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# 4. Data Types
print(df_train.dtypes)

# 5. Feature Analysis (Numerical and Categorical)
numerical_cols = df_train.select_dtypes(include=['number']).columns
categorical_cols = df_train.select_dtypes(include=['object']).columns

for col in numerical_cols:
    if col != 'id' and col != 'Listening_Time_minutes':  # Exclude ID and target
        display(df_train[col].describe())
        plt.figure(figsize=(10, 6))
        sns.histplot(df_train[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

for col in categorical_cols:
    display(df_train[col].value_counts())
    plt.figure(figsize=(10,6))
    df_train[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.show()

# 6. Relationships with Target Variable
for col in numerical_cols:
  if col != 'id' and col != 'Listening_Time_minutes':
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=col, y='Listening_Time_minutes', data=df_train)
    plt.title(f'{col} vs Listening_Time_minutes')
    plt.show()

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y='Listening_Time_minutes', data=df_train)
    plt.title(f'Listening Time by {col}')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# 7. Correlation Analysis
plt.figure(figsize=(12, 10))
sns.heatmap(df_train[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 8. Outlier Detection
for col in numerical_cols:
    plt.figure(figsize=(10,6))
    sns.boxplot(y=df_train[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# Missing Value Imputation
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    df_train[col] = df_train[col].fillna(df_train[col].median())
    df_test[col] = df_test[col].fillna(df_test[col].median())
df_train['Number_of_Ads'] = df_train['Number_of_Ads'].fillna(df_train['Number_of_Ads'].mode()[0])
df_test['Number_of_Ads'] = df_test['Number_of_Ads'].fillna(df_test['Number_of_Ads'].mode()[0])


# Variáveis numéricas com mediana
num_median_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage']
for col in num_median_cols:
    median = df_train[col].median()
    df_train[col].fillna(median, inplace=True)
    df_test[col].fillna(median, inplace=True)

# Variáveis discretas/categóricas com moda
cat_mode_cols = ['Number_of_Ads']
for col in cat_mode_cols:
    mode = df_train[col].mode()[0]
    df_train[col].fillna(mode, inplace=True)
    df_test[col].fillna(mode, inplace=True)

# --- Outlier Handling ---

numerical_cols = df_train.select_dtypes(include=['number']).columns
outlier_limits = {}

for col in numerical_cols:
    if col in ['id', 'Listening_Time_minutes']:
        continue
    
    if col == 'Number_of_Ads':
        upper = df_train[col].quantile(0.95)
        df_train[col] = df_train[col].clip(upper=upper)
        df_test[col] = df_test[col].clip(upper=upper)
        outlier_limits[col] = (None, upper)
    else:
        lower = df_train[col].quantile(0.05)
        upper = df_train[col].quantile(0.95)
        df_train[col] = df_train[col].clip(lower=lower, upper=upper)
        df_test[col] = df_test[col].clip(lower=lower, upper=upper)
        outlier_limits[col] = (lower, upper)

# --- Data Type Consistency ---

df_train['Number_of_Ads'] = df_train['Number_of_Ads'].astype(int)
df_test['Number_of_Ads'] = df_test['Number_of_Ads'].astype(int)

# --- Verification and Assertions ---

print("Missing values in df_train:\n", df_train.isnull().sum())
print("\nMissing values in df_test:\n", df_test.isnull().sum())
print("\nData types in df_train:\n", df_train.dtypes)
print("\nData types in df_test:\n", df_test.dtypes)

# Garantir que não há valores faltantes
assert df_train.isnull().sum().sum() == 0, "Ainda há NaNs em df_train"
assert df_test.isnull().sum().sum() == 0, "Ainda há NaNs em df_test"


print(df_train.columns)



import pandas as pd
from sklearn.preprocessing import StandardScaler

# Evita reprocessamento caso já tenha aplicado o one-hot encoding
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
existing_categorical_cols = [col for col in categorical_cols if col in df_train.columns]

for col in existing_categorical_cols:
    dummies_train = pd.get_dummies(df_train[col], prefix=col)
    dummies_test = pd.get_dummies(df_test[col], prefix=col)

    dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)

    df_train = pd.concat([df_train, dummies_train], axis=1)
    df_test = pd.concat([df_test, dummies_test], axis=1)

# --- Feature scaling ---
numerical_cols_to_scale = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                           'Guest_Popularity_percentage', 'Number_of_Ads']

scaler = StandardScaler()
df_train[numerical_cols_to_scale] = scaler.fit_transform(df_train[numerical_cols_to_scale])
df_test[numerical_cols_to_scale] = scaler.transform(df_test[numerical_cols_to_scale])

# --- Remoção de colunas não utilizadas ---
cols_to_drop = text_cols + categorical_cols
df_train.drop(columns=[col for col in cols_to_drop if col in df_train.columns], inplace=True)
df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns], inplace=True)


# --- Verificações ---
print("Shape of df_train:", df_train.shape)
print("Shape of df_test:", df_test.shape)

print("\nMissing values in df_train:", df_train.isnull().sum().sum())
print("Missing values in df_test:", df_test.isnull().sum().sum())


from sklearn.model_selection import train_test_split

# Separação dos dados (sem estratificação para variáveis contínuas)
X = df_train.drop(['Listening_Time_minutes', 'id'], axis=1)
y = df_train['Listening_Time_minutes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



import xgboost as xgb
import numpy as np

# Instantiate the XGBRegressor model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np

# Definição da grade de hiperparâmetros com amostragem aleatória
param_dist = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Inicialização do modelo XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)

# RandomizedSearchCV com 10 combinações aleatórias e 3 folds
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10,  # Número de combinações testadas
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=0,
    n_jobs=-1,
    random_state=42
)

# Ajuste do RandomizedSearchCV aos dados de validação
random_search.fit(X_val, y_val)

# Avaliação do melhor modelo
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# Resultados
print("Best hyperparameters:", random_search.best_params_)
print("Validation RMSE:", rmse)



y_pred = best_xgb_model.predict(df_test.drop('id', axis=1))
submission_df = pd.DataFrame({'id': df_test['id'], 'Listening_Time_minutes': y_pred})
display(submission_df.head())
submission_df.to_csv('submission.csv', index=False)


from sklearn.metrics import mean_squared_error
import numpy as np


# Retrain the model without 'id'
best_xgb_model.fit(X_train, y_train)

# Predict on the validation set
y_pred_val = best_xgb_model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE after retraining: {rmse_val}")

# Predict on the test set
y_pred_test = best_xgb_model.predict(df_test.drop('id', axis=1))

# Create the submission dataframe
submission_df = pd.DataFrame({'id': df_test['id'], 'Listening_Time_minutes': y_pred_test})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

