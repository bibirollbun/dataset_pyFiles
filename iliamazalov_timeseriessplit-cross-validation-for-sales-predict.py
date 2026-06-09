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
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# === 1) Загрузка и merge ===
train = pd.read_csv("/kaggle/input/bakery-sales-prediction-summer-2025/train.csv")
test = pd.read_csv("/kaggle/input/bakery-sales-prediction-summer-2025/test.csv")
wetter = pd.read_csv("/kaggle/input/bakery-sales-prediction-summer-2025/wetter.csv")
kiwo = pd.read_csv("/kaggle/input/bakery-sales-prediction-summer-2025/kiwo.csv")


wetter = wetter[(wetter["Datum"] >= train["Datum"].min()) &
                (wetter["Datum"] <= train["Datum"].max())]

df = (train.merge(wetter, on="Datum", how="left")
           .merge(kiwo,   on="Datum", how="left"))



# === 2) Базовые фичи времени и погоды ===

# Универсально: приводим к datetime
df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")

# Фичи по дате
df["month"] = df["Datum"].dt.month
df["dayofweek"] = df["Datum"].dt.dayofweek
df["day"] = df["Datum"].dt.day
df["is_month_end"] = df["Datum"].dt.is_month_end.astype(int)

# Взаимодействия с Warengruppe
for col in ["Temperatur", "Windgeschwindigkeit", "Bewoelkung"]:
    if col in df.columns:
        df[f"{col}_by_group"] = df[col] * df["Warengruppe"]

# === 3) Лаги и скользящие (по группе) — без утечек (shift(1)) ===
df = df.sort_values(["Warengruppe", "Datum"]).reset_index(drop=True)


df["rolling_7_mean"] = (
    df.groupby("Warengruppe")["Umsatz"]
      .shift(1)               # чтобы не подсматривать будущее
      .rolling(7)             # окно 7 дней
      .mean()
)

# Заполняем пропуски (NaN от лагов и rolling)
df = df.fillna(0)



# === 4) Формируем X,y. ===
df = df.sort_values(["Datum", "Warengruppe"]).reset_index(drop=True)
X = df.drop(columns=["Umsatz", "Datum", "id"], errors="ignore")
y = df["Umsatz"]

# Label Encoding для категориальных данных
cat_cols = X.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Препроцессинг
numeric_cols = X.columns.tolist()
preproc = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric_cols)],
                            remainder="drop")

def mape(y_true, y_pred, eps=1e-9):
    denom = np.maximum(np.abs(y_true), eps)
    return (np.abs(y_true - y_pred) / denom).mean()*100

# Кастомный разрез по ДАТАМ
def date_splits(df, n_splits=5, date_col="Datum"):
    dates = np.array(sorted(df[date_col].unique()))
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr_d_idx, va_d_idx in tss.split(dates):
        tr_dates = set(dates[tr_d_idx])
        va_dates = set(dates[va_d_idx])
        tr_idx = df.index[df[date_col].isin(tr_dates)]
        va_idx = df.index[df[date_col].isin(va_dates)]
        yield tr_idx, va_idx



# === 5) CV по датам  ===
test_predictions = []
models = []

for fold, (tr_idx, va_idx) in enumerate(date_splits(df, n_splits=5), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    X_tr = preproc.fit_transform(X_tr)
    X_va = preproc.transform(X_va)

    # УБИРАЕМ логарифмирование - работаем с исходным таргетом!
    model = XGBRegressor(
        n_estimators=10000,
        max_depth=12,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=100
    )
    
    model.fit(
        X_tr, y_tr,  # исходный y_tr, не логарифмированный!
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    
    y_pred = model.predict(X_va)
    models.append(model)

    rmse = mean_squared_error(y_va, y_pred, squared=False)

    mae  = mean_absolute_error(y_va, y_pred)
    r2   = r2_score(y_va, y_pred)
    mp   = mape(y_va.values, y_pred)
    print(f"Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}, MAPE={mp:.2f}%")
    



model_to_inspect = models[-1]

feature_names = numeric_cols 

# Важность признаков
importances = model_to_inspect.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)

# Выводим топ-20
print(fi_df.head(20))

# Визуализация
plt.figure(figsize=(10, 7))
plt.barh(fi_df['feature'].head(20)[::-1], fi_df['importance'].head(20)[::-1])
plt.xlabel("Feature Importance")
plt.title("Top 20 Feature Importances")
plt.show()



test_df = test.copy()
test_df["Datum"] = pd.to_datetime(test_df["Datum"])

# === 1) Базовые временные фичи ===
test_df["month"] = test_df["Datum"].dt.month
test_df["dayofweek"] = test_df["Datum"].dt.dayofweek
test_df["day"] = test_df["Datum"].dt.day
test_df["is_month_end"] = test_df["Datum"].dt.is_month_end.astype(int)

# === 2) Приводим погодные данные к datetime ===
wetter["Datum"] = pd.to_datetime(wetter["Datum"], errors="coerce")
kiwo["Datum"] = pd.to_datetime(kiwo["Datum"], errors="coerce")

# Используем только исторические данные
last_train_date = df["Datum"].max()
wetter_historical = wetter[wetter["Datum"] <= last_train_date]
kiwo_historical = kiwo[kiwo["Datum"] <= last_train_date]

# Мержим тест с погодными данными
test_df = test_df.merge(wetter_historical, on="Datum", how="left")
test_df = test_df.merge(kiwo_historical, on="Datum", how="left")

# === 3) Заполняем пропуски ===
test_df["Temperatur"] = test_df["Temperatur"].fillna(0)
test_df["Windgeschwindigkeit"] = test_df["Windgeschwindigkeit"].fillna(0)
test_df["Bewoelkung"] = test_df["Bewoelkung"].fillna(0)
test_df["Wettercode"] = test_df["Wettercode"].fillna(0)

# === 4) Взаимодействия с Warengruppe ===
for col in ["Temperatur", "Windgeschwindigkeit", "Bewoelkung"]:
    test_df[f"{col}_by_group"] = test_df[col] * test_df["Warengruppe"]

# === 5) Лаги и скользящие фичи без утечек ===
last_7_values = df.groupby("Warengruppe")["Umsatz"].apply(lambda x: x.tail(7).mean()).to_dict()
test_df["rolling_7_mean"] = test_df["Warengruppe"].map(last_7_values).fillna(0)

# === 6) Подготовка финальных данных ===
X_test = test_df.drop(columns=["Datum", "id"], errors="ignore")

# Label Encoding для категориальных
cat_cols_test = X_test.select_dtypes(include=["object", "category"]).columns
for col in cat_cols_test:
    le = LabelEncoder()
    all_values = pd.concat([X[col], X_test[col]], axis=0)
    le.fit(all_values.astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Применяем препроцессинг
X_test_processed = preproc.transform(X_test)

# === 7) Предсказания и масштабирование ===
y_test_preds = np.mean([model.predict(X_test_processed) for model in models], axis=0)

scale_factor = df['Umsatz'].median() / np.median(y_test_preds)
y_test_preds_scaled = y_test_preds * scale_factor

# === 8) Создание submission ===
submission = pd.DataFrame({
    "id": test["id"],
    "umsatz": y_test_preds_scaled
})
submission.to_csv("submission14.csv", index=False)

print("Submission saved to submission14.csv")
print(f"Submission median: {submission['umsatz'].median():.2f}")




