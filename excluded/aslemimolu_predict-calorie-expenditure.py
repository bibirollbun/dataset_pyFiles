import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Verileri yÃ¼kleme
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# 2. Ä°lk 5 satÄ±r
print("ğŸ“Š Ä°lk 5 SatÄ±r (train):")
display(train.head())

print("ğŸ“Š Ä°lk 5 SatÄ±r (test):")
display(test.head())


# 3. SÃ¼tunlar ve veri tipleri
print("ğŸ“„ Veri Tipleri (train):\n", train.dtypes)


# 4. Hedef deÄŸiÅŸken daÄŸÄ±lÄ±mÄ±
plt.figure(figsize=(10, 4))
sns.histplot(train['Calories'], bins=100, kde=True)
plt.title("Calories DaÄŸÄ±lÄ±mÄ± (Hedef DeÄŸiÅŸken)")
plt.xlabel("Calories")
plt.ylabel("Frekans")
plt.show()


# 5. Cinsiyet daÄŸÄ±lÄ±mÄ±
sns.countplot(data=train, x="Sex")
plt.title("Cinsiyet DaÄŸÄ±lÄ±mÄ±")
plt.show()


# 6. SayÄ±sal sÃ¼tun Ã¶zet istatistikleri
print("ğŸ“Œ SayÄ±sal SÃ¼tunlar Ã–zet Ä°statistik:")
display(train.describe())


# 7. Eksik veri kontrolÃ¼
print("ğŸ§¯ Eksik DeÄŸer KontrolÃ¼:")
print(train.isnull().sum())


import numpy as np

# 1. Cinsiyet encode iÅŸlemi
train["Sex_male"] = (train["Sex"] == "male").astype(int)
test["Sex_male"]  = (test["Sex"]  == "male").astype(int)

# 2. Calories sÃ¼tununu log1p ile dÃ¶nÃ¼ÅŸtÃ¼r
train["Calories_log"] = np.log1p(train["Calories"])

# 3. Yeni Ã¶zellikler ekleme
# 3.1 BMI
train["BMI"] = train["Weight"] / ((train["Height"]/100) ** 2)
test["BMI"]  = test["Weight"]  / ((test["Height"]/100) ** 2)

# 3.2 BMR (Mifflinâ€“St Jeor)
sex_offset_train = np.where(train["Sex"] == "male", 5, -161)
train["BMR"] = 10 * train["Weight"] + 6.25 * train["Height"] - 5 * train["Age"] + sex_offset_train

sex_offset_test = np.where(test["Sex"] == "male", 5, -161)
test["BMR"]  = 10 * test["Weight"]  + 6.25 * test["Height"]  - 5 * test["Age"]  + sex_offset_test

# 4. KullanÄ±lacak sÃ¼tunlarÄ± belirle
features = [
    "Sex_male", "Age", "Height", "Weight",
    "Duration", "Heart_Rate", "Body_Temp",
    "BMI", "BMR"
]
target = "Calories_log"

X = train[features]
y = train[target]
X_test = test[features]

# Kontrol amaÃ§lÄ± Ã§Ä±ktÄ±lar
print("âœ… Ã–zellikler (X):")
display(X.head())

print("ğŸ�¯ Hedef DeÄŸiÅŸken (y):")
display(y.head())


pip install optuna --quiet


import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

# 1. Optuna objective with 5â€‘fold CV and early stopping

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "tree_method": "hist"
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_va)
        scores.append(np.sqrt(mean_squared_log_error(y_va, preds)))
    return np.mean(scores)

# 2. Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"ğŸ�¯ Best RMSLE: {study.best_value:.5f}")
print("âœ… Best params:", study.best_params)

# 3. Final model training with best_params
best_params = study.best_params
final_model = XGBRegressor(
    **best_params,
    random_state=42,
    tree_method="hist"
)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
final_model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    early_stopping_rounds=50,
    verbose=True
)

# 4. Validation evaluation
y_va_pred = final_model.predict(X_va)
print(f"ğŸ“‰ Final RMSLE (validation): {np.sqrt(mean_squared_log_error(y_va, y_va_pred)):.5f}")

# 5. Test prediction & submission
y_test_log = final_model.predict(X_test)
y_test_pred = np.expm1(y_test_log)
submission = pd.DataFrame({"id": test["id"], "Calories": y_test_pred})
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file created.")


