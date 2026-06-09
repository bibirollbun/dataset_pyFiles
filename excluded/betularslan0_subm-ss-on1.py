import pandas as pd
import numpy as np
import matplotlib as plt


import warnings
warnings.filterwarnings("ignore")



train=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


train.head()


train.info()


train.describe()


test.head()


test.info()


train.isnull().sum()


train_data_cleaned = train.dropna(subset=['num_sold']).copy()


from sklearn.preprocessing import LabelEncoder


def add_time_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)  # Hafta sonu kontrolü
    return df


train_data_cleaned['date'] = pd.to_datetime(train_data_cleaned['date'])
test['date'] = pd.to_datetime(test['date'])


train_data_cleaned = add_time_features(train_data_cleaned)
test = add_time_features(test)


train_data_cleaned['season'] = train_data_cleaned['month'].apply(lambda x: (x%12 + 3)//3)
test['season'] = test['month'].apply(lambda x: (x%12 + 3)//3)



train_data_cleaned['is_holiday'] = 0
test['is_holiday'] = 0




holidays = ['01-01', '12-25']
train_data_cleaned['is_holiday'] = train_data_cleaned['date'].dt.strftime('%m-%d').isin(holidays).astype(int)
test['is_holiday'] = test['date'].dt.strftime('%m-%d').isin(holidays).astype(int)




categorical_columns = ['country', 'store', 'product']



from sklearn.preprocessing import LabelEncoder

for col in categorical_columns:
    train_data_cleaned[col] = train_data_cleaned[col].astype(str)
    test[col] = test[col].astype(str)







combined_data = pd.concat([train_data_cleaned[categorical_columns], test[categorical_columns]], axis=0)
label_encoders = {}  
# Label Encoding'i yeniden uygulama
for col in categorical_columns:
    le = LabelEncoder()
    combined_data[col] = le.fit_transform(combined_data[col])
    label_encoders[col] = le


train_data_cleaned[categorical_columns] = combined_data.iloc[:len(train_data_cleaned)][categorical_columns]
test[categorical_columns] = combined_data.iloc[len(train_data_cleaned):][categorical_columns]




train_data_cleaned['log_num_sold'] = np.log1p(train_data_cleaned['num_sold'])



import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Girdi ve hedef değişkenlerin belirlenmesi
features = ['year', 'month', 'day', 'weekday', 'is_weekend', 'is_holiday', 'country', 'store', 'product']
target = 'log_num_sold'

X = train_data_cleaned[features]
y = train_data_cleaned[target]

# Eğitim ve doğrulama setlerine ayırma
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna ile hiperparametre optimizasyonu
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
    }

    model = LGBMRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    return mae

# Optuna çalıştırma
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# En iyi hiperparametrelerle LightGBM modeli eğitme
best_params = study.best_params
print("Best parameters:", best_params)

lgbm_model = LGBMRegressor(random_state=42, **best_params)
lgbm_model.fit(X_train, y_train)

# Doğrulama setinde tahmin yapma
y_pred = lgbm_model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print(f"Optimized Validation MAE: {mae:.4f}")

# Test setinde tahminler
test['log_num_sold'] = lgbm_model.predict(test[features])
test['num_sold'] = np.expm1(test['log_num_sold'])  # Log dönüşümünü geri al



y_train_pred = lgbm_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Eğitim ve doğrulama hata oranlarını karşılaştıralım
print(f"Training MAE: {train_mae:.4f}")
print(f"Validation MAE: {mae:.4f}")


# Tahmin sonuçlarını Kaggle gönderim formatına uygun şekilde kaydetme
submission = test[['id', 'num_sold']]
submission_file_path = 'submission.csv'  # Dosya yolunu tanımlayın
submission.to_csv(submission_file_path, index=False)

# Kullanıcıya kaydedilen dosyanın yolunu bildirme
print(f"Submission file has been saved to: {submission_file_path}")



submission




