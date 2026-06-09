import torch

print("using", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb

df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


df.head(10)


df.info()


df.isnull().sum()


df.describe().T


df.describe(include="object").T


df["Sex"].unique()


import matplotlib.pyplot as plt
import seaborn as sns

numeric_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

sns.set(style="whitegrid")
plt.figure(figsize=(7, 10))

for i, col in enumerate(numeric_columns):
    plt.subplot(len(numeric_columns), 1, i + 1)
    sns.boxplot(x=df[col], color="skyblue")
    plt.title(f'Boxplot of {col}', fontsize=12)
    plt.xlabel('')
    plt.tight_layout()

plt.show()


df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})


def clip_outliers_iqr(df, columns):
    df_clipped = df.copy()
    for col in columns:
        Q1 = df_clipped[col].quantile(0.25)
        Q3 = df_clipped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clipped[col] = df_clipped[col].clip(lower=lower_bound, upper=upper_bound)
    return df_clipped

outlier_columns = ['Age', "Height", 'Weight', 'Heart_Rate', 'Body_Temp']

df = clip_outliers_iqr(df, outlier_columns)

print("Outliers are clipped to the IQR limits.")


test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test_df.head()


test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})


test_df.info()


test_df.describe().T


"""numeric_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

sns.set(style="whitegrid")
plt.figure(figsize=(7, 10))

for i, col in enumerate(numeric_columns):
    plt.subplot(len(numeric_columns), 1, i + 1)
    sns.boxplot(x=test_df[col], color="skyblue")
    plt.title(f'Boxplot of {col}', fontsize=12)
    plt.xlabel('')
    plt.tight_layout()

plt.show()"""


"""def clip_outliers_iqr(df, columns):
    df_clipped = test_df.copy()
    for col in columns:
        Q1 = df_clipped[col].quantile(0.25)
        Q3 = df_clipped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clipped[col] = df_clipped[col].clip(lower=lower_bound, upper=upper_bound)
    return df_clipped

outlier_columns = ['Age', "Height", 'Weight', 'Heart_Rate', 'Body_Temp']

test_df = clip_outliers_iqr(df, outlier_columns)

print("Outliers are clipped to the IQR limits.")"""


df['BMI'] = df['Weight'] / (df['Height']/100)**2
df['Effort'] = df['Duration'] * df['Heart_Rate']

test_df['BMI'] = test_df['Weight'] / (test_df['Height']/100)**2
test_df['Effort'] = test_df['Duration'] * test_df['Heart_Rate']


X = df.drop(columns=["id", "Calories"])
y = df["Calories"]
X_test_final = test_df.drop(columns=[col for col in ["id", "Calories"] if col in test_df.columns])


# Train test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_model.fit(X_train, y_train)


y_val_pred = xgb_model.predict(X_val)
y_val_pred = np.maximum(y_val_pred, 1e-6)
rmsle = mean_squared_log_error(y_val, y_val_pred) ** 0.5
print(f"Validation RMSLE: {rmsle:.4f}")

y_test_pred = xgb_model.predict(X_test_final)
y_test_pred = np.maximum(y_test_pred, 1e-6)


"""
submission = pd.DataFrame({
    "id": test_df["id"],
    "Calories": y_test_pred.flatten()
})

try:
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("submission.csv is saved!")
except:
    print("An exception occurred!")
"""


import optuna
from sklearn.model_selection import cross_val_score, KFold
def objective(trial):
    params = {
        'device': 'cuda',
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1e-1, log=True),
        'n_estimators': 1000
    }
    
    model = xgb.XGBRegressor(**params)

    y_shifted = y - y.min() + 1

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y_shifted, cv=cv,
                             scoring=lambda est, X, y: -np.sqrt(mean_squared_log_error(y, est.predict(X))))
    return -scores.mean()

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=10)

print("Best params:", study.best_params)





import optuna
from sklearn.model_selection import cross_val_score, KFold

# RMSLE skoru (negatif tahminleri sıfıra sabitler)
def rmsle_score(estimator, X, y):
    y_pred = estimator.predict(X)
    y_pred = np.clip(y_pred, 0, None)  # Negatif tahminleri sıfıra sabitle
    return -np.sqrt(mean_squared_log_error(y, y_pred))  # Skoru minimize edeceğimiz için negatif alındı

def objective(trial):
    # Hyperparameters to be tuned
    params = {
        'device': 'cuda',
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1e-1, log=True),
        'n_estimators': 1000
    }

    # Instantiate the model
    model = xgb.XGBRegressor(**params)

    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # RMSLE score (negatif tahminleri sıfırlayarak)
    scores = cross_val_score(model, X, y, cv=cv, scoring=rmsle_score)
    return -scores.mean()  # minimize edilecek skor

# Optuna çalıştırma
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)

print("Best params:", study.best_params)



best_params = study.best_params

best_model = xgb.XGBRegressor(**best_params, n_estimators=1000)


best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test_final)


submission = test_df[['id']].copy()

submission['Calories'] = y_pred

submission.to_csv('submission.csv', index=False)


print("y_test_pred boyutu:", len(y_test_pred))
print("Submission DataFrame boyutu:", len(submission))

