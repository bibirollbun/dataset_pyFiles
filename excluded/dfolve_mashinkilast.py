import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
from supplemental_english import REGION_CODES, GOVERNMENT_CODES

train = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
submission = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv")




def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)
    )



def extract_features(df):
    df = df.copy()
    df["letters"] = df["plate"].str.extract(r"([A-ZА-Я]{1})[0-9]{3}([A-ZА-Я]{2})")[0] + \
                    df["plate"].str.extract(r"([A-ZА-Я]{1})[0-9]{3}([A-ZА-Я]{2})")[1]
    df["digits"] = df["plate"].str.extract(r"[A-ZА-Я]{1}([0-9]{3})[A-ZА-Я]{2}")[0].astype(int)
    df["region"] = df["plate"].str.extract(r"([0-9]{2,3})$")[0]
    df["region_code"] = pd.to_numeric(df["region"], errors="coerce")

    # Определяю красоту номера
    def is_beautiful(d):
        s = str(d).zfill(3)
        return int(len(set(s)) == 1 or s in ['001', '007', '123', '321', '777', '888', '999'])

    df["is_beautiful"] = df["digits"].apply(is_beautiful)
    df["is_government"] = df["plate"].isin(GOVERNMENT_CODES).astype(int)
    
    # Буквы как отдельные признаки
    df["letter_1"] = df["plate"].str[0]
    df["letter_2"] = df["plate"].str[4]
    df["letter_3"] = df["plate"].str[5]

    # Дата на месяц и год
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df



train = extract_features(train)
test = extract_features(test)

# Признаки и цель
features = ["digits", "region_code", "is_beautiful", "is_government",
            "letter_1", "letter_2", "letter_3", "year", "month"]
target = "price"

# Категориальные признаки
cat_features = ["letter_1", "letter_2", "letter_3", "region_code", "year", "month"]



train["log_price"] = np.log1p(train[target])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
    y_train, y_val = train.iloc[train_idx]["log_price"], train.iloc[val_idx]["log_price"]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    test_pool = Pool(test[features], cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        task_type="GPU",
        random_state=42
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    
    oof_preds[val_idx] = model.predict(val_pool)
    test_preds += model.predict(test_pool) / kf.n_splits





# SMAPE на тренировочных данных 
train_smape = smape(train[target], np.expm1(oof_preds))
print(f"SMAPE on train (CV): {train_smape:.2f}%")

# Сохранение результата
submission["price"] = np.expm1(test_preds).round().astype(int)
submission.to_csv("submission.csv", index=False)
print("Сохранено")


