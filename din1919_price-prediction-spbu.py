import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import optuna
import lightgbm as lgb


try:
    # ĞŸÑƒÑ‚ÑŒ Ğ´Ğ»Ñ� Kaggle
    train = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")
except FileNotFoundError:
    # Ğ›Ğ¾ĞºĞ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ğ¿ÑƒÑ‚ÑŒ
    train = pd.read_csv("./train.csv")
    test = pd.read_csv("./test.csv")

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ id Ğ´Ğ»Ñ� submission
test_ids = test['id']

# Ğ£Ğ´Ğ°Ğ»Ñ�ĞµĞ¼ Ñ†ĞµĞ»ĞµĞ²ÑƒÑ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½ÑƒÑ� Ğ¸Ğ· train
y_train = train['price']
y_train = np.log1p(y_train)  # Ğ›Ğ¾Ğ³Ğ°Ñ€Ğ¸Ñ„Ğ¼Ğ¸Ñ€ÑƒĞµĞ¼ Ğ´Ğ»Ñ� Ğ±Ğ¾Ğ»ĞµĞµ Ğ½Ğ¾Ñ€Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ�

X_train = train.drop(columns=['price', 'id'])
X_test = test.drop(columns=['id'])


# Ğ£Ğ´Ğ°Ğ»ĞµĞ½Ğ¸Ğµ Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¾Ğ² Ğ¿Ğ¾ Ğ¿Ñ€Ğ°Ğ²Ğ¸Ğ»Ñƒ Ñ‚Ñ€Ñ‘Ñ… Ñ�Ğ¸Ğ³Ğ¼ (Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ ĞµÑ�Ğ»Ğ¸ X_train ĞµÑ‰Ñ‘ Ğ½Ğµ Ğ¾Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°Ğ½)
z_scores = np.abs((y_train - y_train.mean()) / y_train.std())
outlier_mask = z_scores < 3
X_train = X_train[outlier_mask]
y_train = y_train[outlier_mask]


def extract_engine_features(engine):
    if not isinstance(engine, str):
        return pd.Series([0, 0])
    power = re.findall(r"(\d+\.?\d*)HP", engine)
    volume = re.findall(r"(\d+\.?\d*)L", engine)
    return pd.Series([float(power[0]) if power else 0, float(volume[0]) if volume else 0])

X_train[['engine_power', 'engine_volume']] = X_train['engine'].apply(extract_engine_features)
X_test[['engine_power', 'engine_volume']] = X_test['engine'].apply(extract_engine_features)

X_train = X_train.drop(columns=['engine'])
X_test = X_test.drop(columns=['engine'])


# Ğ¡Ğ»Ğ¾Ğ²Ğ°Ñ€ÑŒ Ğ¿Ğ¾Ğ¿ÑƒĞ»Ñ�Ñ€Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ñ†Ğ²ĞµÑ‚Ğ¾Ğ² (US/EU Ñ€Ñ‹Ğ½Ğ¾Ğº)
color_popularity = {
    'White': 0.24,
    'Black': 0.20,
    'Gray': 0.16,
    'Silver': 0.12,
    'Blue': 0.09,
    'Red': 0.08,
    'Brown': 0.05,
    'Green': 0.03,
    'Yellow': 0.02,
    'Beige': 0.02,
    'Other': 0.07
}

def get_color_popularity(color):
    return color_popularity.get(str(color).strip(), color_popularity['Other'])

X_train['color_popularity'] = X_train['ext_col'].apply(get_color_popularity)
X_test['color_popularity'] = X_test['ext_col'].apply(get_color_popularity)


# Ğ£Ğ±ĞµĞ´Ğ¸Ğ¼Ñ�Ñ�, Ñ‡Ñ‚Ğ¾ Ğ²Ñ�Ñ‘ Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°ĞµÑ‚ ĞºĞ¾Ñ€Ñ€ĞµĞºÑ‚Ğ½Ğ¾
print("ĞšĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ Ğ² X_train:", list(X_train.columns))
print("ĞšĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ Ğ² X_test:", list(X_test.columns))

# Ğ˜Ñ�Ğ¿Ñ€Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ Ğ¾ÑˆĞ¸Ğ±ĞºÑƒ Ğ² Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ğ¸ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ Ğ¿Ñ€Ğ¾Ğ±ĞµĞ³Ğ°
X_train.rename(columns={'milage': 'mileage'}, inplace=True)
X_test.rename(columns={'milage': 'mileage'}, inplace=True)

# ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ñ�Ğ½Ğ¾Ğ²Ğ°
print("ĞšĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ Ğ² X_train Ğ¿Ğ¾Ñ�Ğ»Ğµ Ğ¸Ñ�Ğ¿Ñ€Ğ°Ğ²Ğ»ĞµĞ½Ğ¸Ñ�:", list(X_train.columns))
print("ĞšĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸ Ğ² X_test Ğ¿Ğ¾Ñ�Ğ»Ğµ Ğ¸Ñ�Ğ¿Ñ€Ğ°Ğ²Ğ»ĞµĞ½Ğ¸Ñ�:", list(X_test.columns))

# Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµĞ¼ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞ³Ğ¾Ğ´Ğ¾Ğ²Ğ¾Ğ¹ Ğ¿Ñ€Ğ¾Ğ±ĞµĞ³
current_year = 2023  # Ğ¢ĞµĞºÑƒÑ‰Ğ¸Ğ¹ Ğ³Ğ¾Ğ´
X_train['car_age'] = current_year - X_train['model_year'] + 1
X_test['car_age'] = current_year - X_test['model_year'] + 1

X_train['annual_mileage'] = X_train['mileage'] / X_train['car_age']
X_test['annual_mileage'] = X_test['mileage'] / X_test['car_age']

# ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ñ‹
print("ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½Ğ½Ğ¾Ğ³Ğ¾ annual_mileage Ğ² X_train:")
print(X_train[['mileage', 'car_age', 'annual_mileage']].head())



# Ğ‘Ñ€ĞµĞ½Ğ´Ñ‹ Ğ¸ Ğ¸Ñ… ÑƒÑ�Ğ»Ğ¾Ğ²Ğ½Ñ‹Ğµ Ñ€ĞµĞ¹Ñ‚Ğ¸Ğ½Ğ³Ğ¸ Ğ½Ğ°Ğ´Ñ‘Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸
brand_reliability = {
    'Toyota': 9.6,
    'Lexus': 9.5,
    'Honda': 9.3,
    'Porsche': 9.1,
    'Subaru': 8.9,
    'Mazda': 8.7,
    'Kia': 8.5,
    'Hyundai': 8.6,
    'Chevrolet': 7.6,
    'Ford': 8.0,
    'BMW': 7.8,
    'Nissan': 8.2,
    'Mercedes-Benz': 7.5,
    'Audi': 7.4,
    'Volkswagen': 6.5,
    'Dodge': 7.2,
    'Volvo': 7.0,
    'Buick': 7.8,
    'Jaguar': 5.2,
    'Land Rover': 5.0,
    'Tesla': 6.0,
    'Ram': 7.5,
    'Jeep': 7.7,
    'Cadillac': 7.0,
    'Chrysler': 6.8,
    'Lamborghini': 5.5,
    'Bentley': 4.5,
    'Rolls-Royce': 4.0,
    'Maserati': 5.3,
    'Alfa Romeo': 5.5,
    'Acura': 7.2,
    'INFINITI': 6.8,
    'Fiat': 5.5,
    'Saab': 6.0,
    'Scion': 7.0,
    'Suzuki': 6.8,
    'Porsche': 9.1,
    'MINI': 7.5,
    'Lincoln': 7.0,
    'GMC': 7.4,
    'Sedici': 6.8,
    'Genesis': 8.0,
    'Lucid': 6.5,
    'Rivian': 6.8,
    'Others': 7.0
}

def get_brand_reliability(brand):
    return brand_reliability.get(str(brand).strip(), brand_reliability['Others'])

X_train['brand_reliability'] = X_train['brand'].apply(get_brand_reliability)
X_test['brand_reliability'] = X_test['brand'].apply(get_brand_reliability)


cat_features = [
    'brand', 'model', 'fuel_type', 'transmission',
    'ext_col', 'int_col', 'accident', 'clean_title'
]

for col in cat_features:
    le = LabelEncoder()
    unique_values = pd.concat([X_train[col].astype(str), X_test[col].astype(str)]).unique()
    le.fit(unique_values)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


low_info_columns = ['clean_title']  # Ğ¸Ğ»Ğ¸ Ğ´Ñ€ÑƒĞ³Ğ¸Ğµ Ğ¼ĞµĞ½ĞµĞµ Ğ²Ğ°Ğ¶Ğ½Ñ‹Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸
X_train = X_train.drop(columns=low_info_columns, errors='ignore')
X_test = X_test.drop(columns=low_info_columns, errors='ignore')


# Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ²Ñ�ĞµÑ… Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² (Ğ²ĞºĞ»Ñ�Ñ‡Ğ°Ñ� Ğ½Ğ¾Ğ²Ñ‹Ğµ)
num_features = [c for c in X_train.columns if X_train[c].dtype != 'O' and c in X_test.columns]

# Ğ—Ğ°Ñ‰Ğ¸Ñ‚Ğ° Ğ¾Ñ‚ inf: Ğ·Ğ°Ğ¼ĞµĞ½Ğ° Ğ´ĞµĞ»ĞµĞ½Ğ¸Ñ� Ğ½Ğ° 0
X_train['car_age'] = np.where(X_train['car_age'] <= 0, 1, X_train['car_age'])  # ĞµÑ�Ğ»Ğ¸ Ğ²Ğ¾Ğ·Ñ€Ğ°Ñ�Ñ‚ <= 0, Ñ�Ñ‚Ğ°Ğ²Ğ¸Ğ¼ 1
X_test['car_age'] = np.where(X_test['car_age'] <= 0, 1, X_test['car_age'])

# ĞŸĞµÑ€ĞµÑ�Ñ‡Ğ¸Ñ‚Ñ‹Ğ²Ğ°ĞµĞ¼ annual_mileage, Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ ÑƒĞ±ĞµĞ´Ğ¸Ñ‚ÑŒÑ�Ñ�
X_train['annual_mileage'] = X_train['mileage'] / X_train['car_age']
X_test['annual_mileage'] = X_test['mileage'] / X_test['car_age']

# Ğ—Ğ°Ğ¼ĞµĞ½Ğ° inf/-inf Ğ½Ğ° NaN, Ğ·Ğ°Ñ‚ĞµĞ¼ Ğ·Ğ°Ğ¿Ğ¾Ğ»Ğ½Ñ�ĞµĞ¼ Ğ¼ĞµĞ´Ğ¸Ğ°Ğ½Ğ¾Ğ¹
from sklearn.impute import SimpleImputer

X_train[num_features] = X_train[num_features].replace([np.inf, -np.inf], np.nan)
X_test[num_features] = X_test[num_features].replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy='median')
X_train[num_features] = imputer.fit_transform(X_train[num_features])
X_test[num_features] = imputer.transform(X_test[num_features])

# Ğ¢ĞµĞ¿ĞµÑ€ÑŒ Ğ¼Ğ¾Ğ¶Ğ½Ğ¾ Ğ±ĞµĞ·Ğ¾Ğ¿Ğ°Ñ�Ğ½Ğ¾ Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½Ñ�Ñ‚ÑŒ StandardScaler
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


# Ğ£Ğ±ĞµĞ¶Ğ´Ğ°ĞµĞ¼Ñ�Ñ�, Ñ‡Ñ‚Ğ¾ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ Ñ�Ğ¾Ğ´ĞµÑ€Ğ¶Ğ°Ñ‚ Ñ‚Ğµ Ğ¶Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸, Ğ² Ñ‚Ğ¾Ğ¼ Ğ¶Ğµ Ğ¿Ğ¾Ñ€Ñ�Ğ´ĞºĞµ
X_test = X_test[X_train.columns]


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        # 'verbose' - Ğ½Ğµ Ğ¿ĞµÑ€ĞµĞ´Ğ°ĞµĞ¼ Ğ·Ğ´ĞµÑ�ÑŒ!
    }

    model = LGBMRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    rmse_scores = []

    for train_idx, valid_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),  # verbose Ğ·Ğ´ĞµÑ�ÑŒ Ğ´Ğ¾Ğ¿ÑƒÑ�Ñ‚Ğ¸Ğ¼
                lgb.log_evaluation(period=0)  # Ğ¿ĞµÑ€Ğ¸Ğ¾Ğ´ Ğ²Ñ‹Ğ²Ğ¾Ğ´Ğ° = 0 â€” Ğ¾Ñ‚ĞºĞ»Ñ�Ñ‡Ğ¸Ñ‚ÑŒ Ğ²Ñ‹Ğ²Ğ¾Ğ´
            ]
        )

        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(pred)))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3)  # Ğ¼Ğ¾Ğ¶Ğ½Ğ¾ ÑƒĞ²ĞµĞ»Ğ¸Ñ‡Ğ¸Ñ‚ÑŒ Ñ‡Ğ¸Ñ�Ğ»Ğ¾ trials Ğ´Ğ»Ñ� Ğ»ÑƒÑ‡ÑˆĞµĞ³Ğ¾ Ğ¿Ğ¾Ğ¸Ñ�ĞºĞ°

best_params = study.best_params

print("Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:", best_params)


# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ n_estimators Ğ¸Ğ· best_params, Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ Ğ¸Ğ·Ğ±ĞµĞ¶Ğ°Ñ‚ÑŒ Ğ´ÑƒĞ±Ğ»Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ñ�
if 'n_estimators' in best_params:
    del best_params['n_estimators']

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ñ‘Ğ¼ final_model Ñ� Ğ»ÑƒÑ‡ÑˆĞ¸Ğ¼Ğ¸ Ğ³Ğ¸Ğ¿ĞµÑ€Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ°Ğ¼Ğ¸ Ğ¸ Ñ„Ğ¸ĞºÑ�Ğ¸Ñ€ÑƒĞµĞ¼ n_estimators
final_model = LGBMRegressor(**best_params, n_estimators=3000, random_state=42)

# Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ñ� ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸ĞµĞ¹
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    print(f"\nFold {fold + 1}")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    oof_preds[valid_idx] = final_model.predict(X_val)
    test_preds += final_model.predict(X_test) / FOLDS
    
    val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(oof_preds[valid_idx])))
    print(f"Fold {fold + 1} RMSE: {val_rmse:.2f}")


overall_rmse = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof_preds)))
print(f"\nĞ�Ğ±Ñ‰Ğ¸Ğ¹ RMSE Ğ½Ğ° ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸: {overall_rmse:.2f}")


# ĞŸÑ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ğ¾Ğµ Ğ¿Ñ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğº Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ�Ğ¼
final_prices = np.expm1(test_preds)

# Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ DataFrame
submission = pd.DataFrame({'id': test_ids, 'price': final_prices})

# Ğ¯Ğ²Ğ½Ğ¾ Ğ¿Ñ€Ğ¸Ğ²Ğ¾Ğ´Ğ¸Ğ¼ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºÑƒ price Ğº Ñ‚Ğ¸Ğ¿Ñƒ float
submission['price'] = submission['price'].astype(float)

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ² CSV Ğ±ĞµĞ· Ğ¸Ğ½Ğ´ĞµĞºÑ�Ğ°
submission.to_csv("submission.csv", index=False)

print("Ğ¤Ğ°Ğ¹Ğ» submission.csv Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½!")

