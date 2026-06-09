import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


train_path = "/kaggle/input/russian-car-plates-prices-prediction/train.csv"
test_path = "/kaggle/input/russian-car-plates-prices-prediction/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day_of_week"] = train["date"].dt.dayofweek

test["year"] = test["date"].dt.year
test["month"] = test["date"].dt.month
test["day_of_week"] = test["date"].dt.dayofweek


def process_plate(plate):
    letters = "".join([char for char in plate if char.isalpha()]) 
    digits = "".join([char for char in plate if char.isdigit()]) 
    region = int(digits[-2:]) 
    return letters, digits[:3], region
train[["letters", "digits", "region"]] = train["plate"].apply(lambda x: pd.Series(process_plate(x)))
test[["letters", "digits", "region"]] = test["plate"].apply(lambda x: pd.Series(process_plate(x)))


letters_mean_price = train.groupby("letters")["price"].mean()
train["letters_te"] = train["letters"].map(letters_mean_price)
test["letters_te"] = test["letters"].map(letters_mean_price).fillna(train["price"].mean())  # Замена NaN

digits_counts = train["digits"].value_counts()
train["digits_freq"] = train["digits"].map(digits_counts)
test["digits_freq"] = test["digits"].map(digits_counts).fillna(1)

popular_regions = train["region"].value_counts().index[:20]
train["region_grouped"] = train["region"].apply(lambda x: x if x in popular_regions else 999)
test["region_grouped"] = test["region"].apply(lambda x: x if x in popular_regions else 999)

plate_counts = train["plate"].value_counts()
train["plate_freq"] = train["plate"].map(plate_counts)
test["plate_freq"] = test["plate"].map(plate_counts).fillna(1)

train["log_price"] = np.log1p(train["price"])


drop_cols = ["id", "plate", "date", "price", "letters", "digits", "region"]
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)


train.to_csv("/kaggle/working/train_prepared.csv", index=False)
test.to_csv("/kaggle/working/test_prepared.csv", index=False)
print(train.head())
print(test.head())


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Функция для расчёта SMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

train_path = "/kaggle/working/train_prepared.csv"
test_path = "/kaggle/working/test_prepared.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = train.drop(columns=["log_price"])
y = train["log_price"]
X_test = test.copy()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
cb_test_pred = np.zeros(len(X_test))
xgb_test_pred = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost
    cb_model = CatBoostRegressor(iterations=5000, learning_rate=0.05, depth=6, loss_function="MAE", verbose=500)
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=500)
    cb_oof[val_idx] = cb_model.predict(X_val)
    cb_test_pred += cb_model.predict(X_test) / kf.n_splits

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=5000, learning_rate=0.05, max_depth=6, eval_metric="mae", objective="reg:absoluteerror")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=500
    )
    xgb_oof[val_idx] = xgb_model.predict(X_val)
    xgb_test_pred += xgb_model.predict(X_test) / kf.n_splits

stack_train = pd.DataFrame({"cb": cb_oof, "xgb": xgb_oof})
stack_test = pd.DataFrame({"cb": cb_test_pred, "xgb": xgb_test_pred})

#(GradientBoostingRegressor)
meta_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4)
meta_model.fit(stack_train, y)

final_pred = meta_model.predict(stack_test)
final_pred_price = np.expm1(final_pred)

submission = pd.DataFrame({"id": test.index, "price": final_pred_price})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(f"SMAPE на валидации: {smape(y, meta_model.predict(stack_train)):.4f}")



sample_submission = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv')

submission = pd.DataFrame({
    "id": sample_submission["id"],
    "price": final_pred_price
})

# Сохраняем финальный сабмит
submission.to_csv("/kaggle/working/submission.csv", index=False)


