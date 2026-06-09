import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
w.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
train.head()


# For train
train['Number_of_Ads'].fillna(train['Number_of_Ads'].mean(),inplace=True)
train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean(),inplace=True)
train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(),inplace=True)
train.drop(columns=['id'], inplace=True,errors='ignore')
# For test
test['Number_of_Ads'].fillna(test['Number_of_Ads'].mean(),inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].mean(),inplace=True)
test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(),inplace=True)
test.drop(columns=['id'], inplace=True) 


num_cols=train.select_dtypes(include=(['int64','float64'])).columns.tolist()
cat_cols=train.select_dtypes(include=(['object'])).columns.tolist()


sns.set(style='whitegrid')
for i in num_cols:
    plt.figure(figsize=(6,5))
    sns.histplot(train[i],kde=True,color='blue',edgecolor='black')
    plt.xlabel(f'Distribution of {i}')
    plt.ylabel('Frequency')
    plt.show()


sns.set(style='whitegrid')
cat_cols_modified=['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for i in cat_cols_modified:
    plt.figure(figsize=(6,5))
    sns.histplot(train[i],kde=True,color='blue',edgecolor='black')
    plt.xlabel(f'Distribution of {i}')
    plt.xticks(rotation=90)
    plt.ylabel('Frequency')
    plt.show()


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Hedefi ayır
target = 'Listening_Time_minutes'
x = train.drop(columns=[target])
y = train[target]

# DOĞRU ŞEKİLDE num_cols ve cat_cols'ları x üzerinden tanımla
num_cols = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = x.select_dtypes(include=['object']).columns.tolist()

# Sonra train_test_split yap
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- 1. Önce Preprocessing Pipelines Tanımla ---
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

col_transformer = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# --- 2. LightGBM Regressor Model Tanımla ---
lg = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    random_state=42,
    verbosity=-1,
    boosting_type='gbdt'
)

# --- 3. Pipeline Kur ---
model = Pipeline([
    ('preprocessor', col_transformer),
    ('regressor', lg)
])

# --- 4. Parametre Alanlarını Belirle (Random Search için) ---
param_distributions = {
    'regressor__max_depth': randint(5, 30),           # 5 ile 30 arası ağaç derinliği
    'regressor__num_leaves': randint(20, 300),         # 20 ile 300 arası yaprak
    'regressor__learning_rate': uniform(0.005, 0.05),  # 0.005 ile 0.055 arası öğrenme oranı
    'regressor__n_estimators': randint(500, 2000),     # 500 ile 2000 arası iterasyon
    'regressor__subsample': uniform(0.6, 0.4),         # 0.6 ile 1.0 arası
    'regressor__colsample_bytree': uniform(0.6, 0.4)   # 0.6 ile 1.0 arası
}

# --- 5. RandomizedSearchCV Kur (Hızlı Tarama Yapacak) ---
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=20,              # 20 farklı kombinasyon dene (çok daha fazla istersen artırılır)
    scoring='neg_root_mean_squared_error',   # RMSE minimize etmeye çalışıyoruz
    cv=3,                   # 3-Fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1               # Tüm işlemcileri kullanarak hızlandır
)

# --- 6. Random Search Fit (En İyi Parametreleri Bul) ---
random_search.fit(x_train, y_train)

# --- 7. En İyi Model ile Sonuçları Yazdır ---
best_model = random_search.best_estimator_

print("Best Parameters Found:", random_search.best_params_)

# Test seti üzerinde tahmin yap
y_pred = best_model.predict(x_test)

# Skorları yazdır
print(f'MSE: {mean_squared_error(y_test, y_pred):.2f}')
print(f'R2 Score: {r2_score(y_test, y_pred) * 100:.2f}')
rmsc = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmsc:.4f}')

# İlk 10 gerçek vs tahmin yazdır
for actual, pred in zip(y_test[:10], y_pred[:10]):
    print(f'Actual: {actual:.2f} | Predicted: {pred:.2f}')



import joblib

# Save the BEST model, not the old one
joblib.dump(best_model, 'lgbm_model.pkl')

print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load('lgbm_model.pkl')

# Now you can use the loaded model for predictions
y_pred_loaded = loaded_model.predict(x_test)
print(f"Predictions using loaded model: {y_pred_loaded[:10]}")




# Never done this step :)
# Re-load the 'id' column from original test file
test_ids = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')['id']

# Generate predictions for the competition test set
test_predictions = best_model.predict(test)

# Prepare submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': test_predictions
})

# Save to CSV for submission
submission.to_csv('submission.csv', index=False)





