import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")



target = "Listening_Time_minutes"
test_ids = test["id"]

# Combine features
combined = pd.concat([train.drop(columns=[target]), test], axis=0).reset_index(drop=True)

# Encode categorical columns
for col in combined.select_dtypes(include="object").columns:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))



X = combined.iloc[:len(train), :]
X_test = combined.iloc[len(train):, :]
y = train[target]



kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"ðŸ“‚ Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=10000,
        learning_rate=0.01,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits



rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"âœ… Overall CV RMSE: {rmse:.4f}")



submission["Listening_Time_minutes"] = test_preds
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")


