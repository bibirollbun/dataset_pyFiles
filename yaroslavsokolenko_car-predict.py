import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
DATA_PATH = "/kaggle/input/russian-car-plates-prices-prediction/"


def extract_number_features(plate: str):
    match = re.search(r'([A-ZĞ�-Ğ¯]+)(\d+)([A-ZĞ�-Ğ¯]+)', plate)
    if match:
        letters1, digits, letters2 = match.groups()
        return letters1, digits, letters2
    return "", "0", ""


def is_beautiful_sequence(x):
    return int(any(sub in x for sub in ["777", "999", "000", "001", "123", "111", "333", "888"]))

def elite_letter_score(plate):
    elite = {"Ğ�", "Ğœ", "Ğ ", "Ğ•", "Ğš", "Ğ¥", "Ğ’", "Ğ�", "Ğ¡", "Ğ�", "Ğ¢"}  # Ğ§Ğ°Ñ�Ñ‚Ğ¾ Ğ²Ñ�Ñ‚Ñ€ĞµÑ‡Ğ°Ñ�Ñ‚Ñ�Ñ� Ğ² ĞºÑ€Ğ°Ñ�Ğ¸Ğ²Ñ‹Ñ… Ğ½Ğ¾Ğ¼ĞµÑ€Ğ°Ñ…
    return sum(1 for ch in plate if ch in elite)

def has_repeating_pair(x):
    return int(any(x[i] == x[i+1] and x[i+1] == x[i+2] for i in range(len(x)-2)))

def engineer_features(df: pd.DataFrame, scaler: StandardScaler = None, is_train: bool = True):
    df = df.dropna(subset=["plate"]).copy()
    df["letters1"], df["digits"], df["letters2"] = zip(*df["plate"].apply(extract_number_features))
    df["digits"] = pd.to_numeric(df["digits"], errors="coerce")
    df["digits"] = df["digits"].fillna(df["digits"].median())

    df["letters1"] = df["letters1"].astype("category").cat.codes
    df["letters2"] = df["letters2"].astype("category").cat.codes

    df["digit_count"] = df["digits"].astype(str).apply(len)
    df["digit_sum"] = df["digits"].astype(str).apply(lambda x: sum(int(d) for d in x if d.isdigit()))
    df["unique_digits"] = df["digits"].astype(str).apply(lambda x: len(set(x)))
    df["unique_letters"] = df["plate"].apply(lambda x: len(set(filter(str.isalpha, x))))
    df["double_letters"] = df["plate"].apply(lambda x: sum(1 for i in range(len(x)-1) if x[i] == x[i+1]))
    df["plate_length"] = df["plate"].apply(len)
    df["letter_count"] = df["plate"].apply(lambda x: sum(c.isalpha() for c in x))
    df["most_common_char_count"] = df["plate"].apply(lambda x: max([x.count(c) for c in set(x)]))
    df["is_palindrome"] = df["plate"].apply(lambda x: int(x == x[::-1]))
    df["is_beautiful"] = df["plate"].apply(lambda x: int(any(x.count(ch) >= 3 for ch in set(x))))

    # ğŸ”¥ Ğ�Ğ¾Ğ²Ñ‹Ğµ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸:
    df["has_beautiful_seq"] = df["plate"].apply(is_beautiful_sequence)
    df["elite_letters"] = df["plate"].apply(elite_letter_score)
    df["repeating_triplet"] = df["plate"].apply(has_repeating_pair)
    df["digit_range"] = df["digits"].astype(str).apply(lambda x: int(max(x)) - int(min(x)) if x.isdigit() and len(x) > 1 else 0)
    df["only_letters_or_digits"] = df["plate"].apply(lambda x: int(x.isalpha() or x.isdigit()))

    features = [
        "letters1", "digits", "letters2", "digit_count", "digit_sum",
        "unique_letters", "unique_digits", "double_letters", "plate_length",
        "letter_count", "most_common_char_count", "is_palindrome", "is_beautiful",
        "has_beautiful_seq", "elite_letters", "repeating_triplet",
        "digit_range", "only_letters_or_digits"
    ]
    
    X = df[features].copy()
    num_cols = [
        "digits", "digit_count", "digit_sum", "unique_digits", "plate_length",
        "letter_count", "most_common_char_count", "digit_range", "elite_letters"
    ]
    if is_train:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    else:
        X[num_cols] = scaler.transform(X[num_cols])

    return X, scaler



def remove_outliers(df: pd.DataFrame, column: str):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] > (Q1 - 1.5 * IQR)) & (df[column] < (Q3 + 1.5 * IQR))].copy()


def train_models(X_train, y_train):
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=1500,
        max_depth=40,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    lgbm_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)

    return xgb_model, rf_model, lgbm_model


def evaluate_models(models, X_valid, y_valid):
    xgb_model, rf_model, lgbm_model = models
    y_pred_xgb = xgb_model.predict(X_valid)
    y_pred_rf = rf_model.predict(X_valid)
    y_pred_lgbm = lgbm_model.predict(X_valid)
    #Ğ°Ğ½Ñ�Ğ°Ğ¼Ğ±Ğ»ÑŒ
    y_pred_ensemble = (y_pred_xgb + y_pred_rf + y_pred_lgbm) / 3
    mae = mean_absolute_error(y_valid, y_pred_ensemble)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred_ensemble))
    print(f"Ğ�Ğ½Ñ�Ğ°Ğ¼Ğ±Ğ»ÑŒ (Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ) MAE: {mae:.2f}")
    print(f"Ğ�Ğ½Ñ�Ğ°Ğ¼Ğ±Ğ»ÑŒ (Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ) RMSE: {rmse:.2f}")
    
    return y_pred_ensemble



train_df = pd.read_csv(DATA_PATH + "train.csv")
train_df = remove_outliers(train_df, "price")
train_df.dropna(subset=["plate", "price"], inplace=True)
train_df["log_price"] = np.log1p(train_df["price"])
X, scaler = engineer_features(train_df, is_train=True)
y = train_df["log_price"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model, rf_model, lgbm_model = train_models(X_train, y_train)
evaluate_models((xgb_model, rf_model, lgbm_model), X_valid, y_valid)
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(lgbm_model, "lgbm_model.pkl")
stack_model = StackingRegressor(
    estimators=[
        ("xgb", xgb_model),
        ("rf", rf_model),
        ("lgbm", lgbm_model)
    ],
    final_estimator=LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42
    )
)
stack_model.fit(X_train, y_train)
y_pred_stack_valid = stack_model.predict(X_valid)
mae_stack = mean_absolute_error(y_valid, y_pred_stack_valid)
rmse_stack = np.sqrt(mean_squared_error(y_valid, y_pred_stack_valid))
print(f"StackingRegressor MAE: {mae_stack:.2f}")
print(f"StackingRegressor RMSE: {rmse_stack:.2f}")
test_df = pd.read_csv(DATA_PATH + "test.csv")
X_test_final, _ = engineer_features(test_df, scaler=scaler, is_train=False)
pred_xgb = xgb_model.predict(X_test_final)
pred_rf = rf_model.predict(X_test_final)
pred_lgbm = lgbm_model.predict(X_test_final)
pred_stack = stack_model.predict(X_test_final)
test_df["price"] = np.expm1(pred_stack)
submission_csv = test_df[["id", "price"]]
submission_csv.to_csv("Submission.csv", index=False)
print("Ğ¤Ğ°Ğ¹Ğ» Submission.csv Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ‘Ğ½!")
submission_print = test_df[["plate", "price"]]
top10_submission = submission.nlargest(10, "price")
print("ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ½Ğ¾Ğ¼ĞµÑ€Ğ° Ğ¸ Ñ†ĞµĞ½Ñ‹ (Ñ‚Ğ¾Ğ¿ 10):")
print(top10_submission)


