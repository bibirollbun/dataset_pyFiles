!pip install optuna



!pip install xgboost



import os
import sys
import gc
import random
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error





import kagglehub
kagglehub.login()


sparta_2024_data_science_competition_path = kagglehub.competition_download('sparta-2024-data-science-competition')

print('Data source import complete.')


import shutil

src = sparta_2024_data_science_competition_path
dst = '/content/sparta-2024-data-science-competition'

shutil.copytree(src, dst)
os.listdir(dst)


TRAIN_PATH = "/content/sparta-2024-data-science-competition/train.csv"
TEST_PATH = "/content/sparta-2024-data-science-competition/test.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)


df_train = df_train.drop_duplicates()
df_test = df_test.drop_duplicates()


df_train.duplicated().sum()
df_test.duplicated().sum()


col_drop = [
    'name', 'description',
    'neighborhood_overview', 'host_id',
    'host_name', 'host_since',
    'host_location', 'host_about',
    'bathrooms_text', 'amenities'
]

df_train = df_train.drop(columns = col_drop)
df_test = df_test.drop(columns = col_drop)


missing_values = df_train.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing Values per Kolom:")
print(missing_values)

missing_percentage = (missing_values / df_train.shape[0]) * 100
print("\nPersentase Missing Values per Kolom:")
print(missing_percentage)


missing_values = df_test.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing Values per Kolom:")
print(missing_values)

missing_percentage = (missing_values / df_test.shape[0]) * 100
print("\nPersentase Missing Values per Kolom:")
print(missing_percentage)


df_clean_train = df_train.fillna(0)
df_clean_test = df_test.fillna(0)
df_clean_test.isnull().sum()


df_train.head()


df_train.describe()


# Distribusi kolom numerik
df_train[['price', 'accommodates', 'bathrooms', 'bedrooms', 'number_of_reviews',	'host_total_listings_count',	'accommodates',	'beds',	'availability_30',
      'availability_60',	'number_of_reviews',	'availability_eoy',	'number_of_reviews_ly',	'estimated_revenue_l365d',	'review_scores_rating']].hist(bins=50, figsize=(15, 10))
plt.tight_layout()
plt.show()


# Visualisasi harga vs kapasitas dan tipe kamar
sns.boxplot(x='room_type', y='price', data = df_train)
plt.title('Price vs Room Type')
plt.show()


plt.figure(figsize=(15, 10))
# Pilih hanya kolom numerik
numerical_data = df_train.select_dtypes(include=['number'])
correlation_matrix = numerical_data.corr()

# Buat heatmap korelasi
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()




label_encoder = LabelEncoder()
categorical_columns = df_train.select_dtypes(include=['object']).columns

for col in df_train.columns:
    if df_train[col].dtype == "object":
        le = LabelEncoder()
        le.fit(list(df_train[col].astype(str)) + list(df_test[col].astype(str)))
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))

print(df_train.dtypes)


X = df_train.drop(columns=['price'])
y = df_train['price']

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
        "random_state": 42,
        "tree_method": "hist",
        "verbosity": 0
    }

    # Buat XGBRegressor model
    model = xgb.XGBRegressor(**params)

    # Gunakan cross-validation untuk mengevaluasi model
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    return score.mean()

# Train final model menggunakan parameter dari optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=75)
print("Best params (XGBoost):", study.best_params)
print("Best RMSE (XGBoost):", -study.best_value)
final_model = xgb.XGBRegressor(**study.best_params)
final_model.fit(X_train, y_train)


from sklearn.metrics import r2_score

params = {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.09661706418039724, 'subsample': 0.7372231358415464, 'colsample_bytree': 0.9754542040968331, 'reg_alpha': 0.1335614207838155, 'reg_lambda': 0.46065505709564925}
model = xgb.XGBRegressor(
    **params,
    random_state=42
)

# Latih model
X_train = X_train.drop(columns = "id")
X_test = X_test.drop(columns = "id")
model.fit(X_train, y_train)

# # Prediksi
y_pred = model.predict(X_test)

# Evaluasi
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")


X_test_predict = df_test.copy()
X_test_predict = X_test_predict.drop(columns= ["id"])

df_test['price'] = model.predict(X_test_predict)

submission = pd.DataFrame({
    'id': df_test['id'],
    'price': df_test['price']
})

submission.to_csv('submission.csv', index=False)
print("Submission saved successfully!")


