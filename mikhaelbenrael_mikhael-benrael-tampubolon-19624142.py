# === [1] Import Libraries ===
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

# Visual setting
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)

# Read files
base_path = '/kaggle/input/sparta-2024-data-science-competition'
train_df = pd.read_csv(f'{base_path}/train.csv', index_col='id')
test_df = pd.read_csv(f'{base_path}/test.csv', index_col='id')

print(f"Train: {train_df.shape}, Test: {test_df.shape}")



# Cek info dasar dan missing values
missing_counts = train_df.isnull().sum()
missing_pct = (missing_counts / len(train_df)).sort_values(ascending=False)

print("Kolom dengan missing terbanyak:")
print(missing_pct[missing_pct > 0][:10])

# Visualisasi distribusi price
sns.histplot(train_df['price'], bins=80, kde=True)
plt.title("Distribusi Harga Sebelum Transformasi")
plt.show()

# Transformasi log
train_df['price_log'] = np.log1p(train_df['price'])

sns.histplot(train_df['price_log'], bins=80, kde=True, color='orange')
plt.title("Distribusi Harga Setelah Log1p")
plt.show()



# Room Type
sns.countplot(data=train_df, x='room_type', order=train_df['room_type'].value_counts().index)
plt.title("Distribusi Room Type")
plt.show()

# Boxplot Harga per Room Type
sns.boxplot(data=train_df, x='room_type', y='price_log')
plt.title("Harga (Log) Berdasarkan Room Type")
plt.show()

# Korelasi antar fitur numerik
corr_matrix = train_df.select_dtypes(include=[np.number]).corr()
corr_target = corr_matrix['price_log'].abs().sort_values(ascending=False)
print("Top korelasi dengan price_log:")
print(corr_target.head(10))



# Hapus kolom yang terlalu banyak missing (>50%)
drop_cols = missing_pct[missing_pct > 50].index.tolist()
df_clean = train_df.drop(columns=drop_cols)

# Bagi fitur numerik dan kategorikal
num_cols = df_clean.select_dtypes(include=['int64', 'float64']).drop(columns=['price', 'price_log']).columns.tolist()
cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

# Buat column transformer
numeric_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))
])

transformer = ColumnTransformer([
    ('num', numeric_transform, num_cols),
    ('cat', categorical_transform, cat_cols)
])



# Split data
X_data = df_clean.drop(columns=['price', 'price_log'])
y_data = df_clean['price_log']

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Model pipeline
model_pipeline = Pipeline([
    ('prep', transformer),
    ('xgb', XGBRegressor(n_jobs=-1, random_state=42, verbosity=0))
])



print(model_pipeline.named_steps)
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 6],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    model_pipeline, 
    param_grid, 
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

print("Tuning model...")
grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")



best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train)
y_pred_valid = best_model.predict(X_valid)

print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("Valid RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred_valid)))

print("Train R²:", r2_score(y_train, y_pred_train))
print("Valid R²:", r2_score(y_valid, y_pred_valid))



# Sesuaikan kolom pada test
test_ready = test_df[X_data.columns.intersection(test_df.columns)]

# Generate prediksi
test_preds = best_model.predict(test_ready)

# Format output
submission = pd.DataFrame({
    'id': test_df.index,
    'price': np.expm1(test_preds)  # Kembalikan dari log1p
})

submission.to_csv("submission.csv", index=False)
print("Submission file created!")


