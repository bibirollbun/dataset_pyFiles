import pandas as pd
import numpy as np

# Пути к данным
train_path = "/kaggle/input/russian-car-plates-prices-prediction/train.csv"
test_path = "/kaggle/input/russian-car-plates-prices-prediction/test.csv"

# Загрузка данных
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# Преобразуем дату в datetime формат
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

# Извлекаем дополнительные признаки из даты
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day_of_week"] = train["date"].dt.dayofweek
train["day_of_month"] = train["date"].dt.day

test["year"] = test["date"].dt.year
test["month"] = test["date"].dt.month
test["day_of_week"] = test["date"].dt.dayofweek
test["day_of_month"] = test["date"].dt.day



# Функция для обработки номера (plate)
def process_plate(plate):
    letters = "".join([char for char in plate if char.isalpha()])  # Буквы
    digits = "".join([char for char in plate if char.isdigit()])  # Цифры
    region = int(digits[-2:])  # Регион (последние 2 цифры)
    return letters, digits[:3], region

# Применяем функцию к каждому номеру
train[["letters", "digits", "region"]] = train["plate"].apply(lambda x: pd.Series(process_plate(x)))
test[["letters", "digits", "region"]] = test["plate"].apply(lambda x: pd.Series(process_plate(x)))


# Target Encoding для букв
letters_mean_price = train.groupby("letters")["price"].mean()
train["letters_te"] = train["letters"].map(letters_mean_price)
test["letters_te"] = test["letters"].map(letters_mean_price).fillna(train["price"].mean())  # Замена NaN

# Частота встречаемости комбинаций цифр
digits_counts = train["digits"].value_counts()
train["digits_freq"] = train["digits"].map(digits_counts)
test["digits_freq"] = test["digits"].map(digits_counts).fillna(1)

# Группировка редких регионов
popular_regions = train["region"].value_counts().index[:20]  # Топ-20 регионов
train["region_grouped"] = train["region"].apply(lambda x: x if x in popular_regions else 999)
test["region_grouped"] = test["region"].apply(lambda x: x if x in popular_regions else 999)

# Логарифмирование цен
train["log_price"] = np.log1p(train["price"])


# Удаление ненужных столбцов
drop_cols = ["id", "plate", "date", "price", "letters", "digits", "region"]
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)

# Выводим информацию о новых признаках
print("Первые строки обработанного train.csv:")
print(train.head())
print("\nПервые строки обработанного test.csv:")
print(test.head())



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


# Отделяем целевую переменную
X = train.drop(columns=["log_price"])
y = train["log_price"]
X_test = test.copy()


# Функция SMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# KFold для Stacking
kf = KFold(n_splits=8, shuffle=True, random_state=42)


# Базовые модели с подобранными гиперпараметрами
xgb_model = xgb.XGBRegressor(
    n_estimators=500, 
    learning_rate=0.0655, 
    max_depth=6, 
    subsample=0.6377, 
    colsample_bytree=0.9866, 
    reg_lambda=0.5495
)

cb_model = CatBoostRegressor(
    iterations=1000, 
    learning_rate=0.1058, 
    depth=10, 
    l2_leaf_reg=0.0103, 
    subsample=0.8279, 
    loss_function="MAE", 
    verbose=500
)

# Level-1 predictions
xgb_oof = np.zeros(len(X))
cb_oof = np.zeros(len(X))
xgb_test_pred = np.zeros(len(X_test))
cb_test_pred = np.zeros(len(X_test))



# Stacking
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=500)
    xgb_oof[val_idx] = xgb_model.predict(X_val)
    xgb_test_pred += xgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=500)
    cb_oof[val_idx] = cb_model.predict(X_val)
    cb_test_pred += cb_model.predict(X_test) / kf.n_splits

# Мета-признаки
stack_train = pd.DataFrame({"xgb": xgb_oof, "cb": cb_oof})
stack_test = pd.DataFrame({"xgb": xgb_test_pred, "cb": cb_test_pred})

# Обучение мета-модели (XGBoost)
meta_model = xgb.XGBRegressor(
    n_estimators=500, 
    learning_rate=0.0655, 
    max_depth=6, 
    subsample=0.6377, 
    colsample_bytree=0.9866, 
    reg_lambda=0.5495
)
meta_model.fit(stack_train, y)


# Финальные предсказания
final_pred = meta_model.predict(stack_test)

# Переводим обратно в цену
final_pred_price = np.expm1(final_pred)

# Вывод результатов
print(f"SMAPE на валидации: {smape(y, meta_model.predict(stack_train)):.4f}")



# 10 самых дорогих номеров из обучающего датасета
top_train_plates = pd.read_csv(train_path).nlargest(10, "price")[["plate", "price"]]
print("Топ-10 самых дорогих номеров (обучающие данные):")
print(top_train_plates)

# 10 самых дорогих номеров из предсказанных данных
test_data = pd.read_csv(test_path)
test_data["predicted_price"] = final_pred_price  # Добавляем предсказанные цены
top_test_plates = test_data.nlargest(10, "predicted_price")[["plate", "predicted_price"]]
print("\nТоп-10 самых дорогих номеров (предсказанные данные):")
print(top_test_plates)


