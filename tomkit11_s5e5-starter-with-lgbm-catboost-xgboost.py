import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 25))
for i, col in enumerate(numeric_cols):
    data = train[col].replace([np.inf, -np.inf], np.nan).dropna()
    sns.histplot(data, bins=50, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

X = train.drop(columns=["id", "Calories"])
y = np.log1p(train["Calories"])
X_test = test.drop(columns=["id"])


kf = KFold(n_splits=5, shuffle=True, random_state=42)


def train_model(ModelClass, model_name, **params):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    print(f"\nTraining {model_name}...")

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        model = ModelClass(**params)
        if model_name == "LightGBM":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse")
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

        oof_preds[valid_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / kf.n_splits

        fold_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    full_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"{model_name} CV RMSE: {full_rmse:.4f}")
    return oof_preds, test_preds


oof_lgb, test_lgb = train_model(LGBMRegressor, "LightGBM", random_state=42)
oof_cat, test_cat = train_model(CatBoostRegressor, "CatBoost", verbose=0, random_state=42)
oof_xgb, test_xgb = train_model(XGBRegressor, "XGBoost", verbosity=0, random_state=42)


test_preds_avg = (test_lgb + test_cat + test_xgb) / 3


submission["Calories"] = np.expm1(test_preds_avg)
submission.to_csv("submission.csv", index=False)
print("\nâœ… Submission file saved.")

