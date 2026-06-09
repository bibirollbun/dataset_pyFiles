# Import libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# Feature Engineering
numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            cross_term_name = f"{numerical_features[i]}_x_{numerical_features[j]}"
            df_new[cross_term_name] = df_new[numerical_features[i]] * df_new[numerical_features[j]]
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)


# Add BMI & Intensity
train["BMI"] = train["Weight"] / (train["Height"] / 100) ** 2
test["BMI"] = test["Weight"] / (test["Height"] / 100) ** 2

train["Intensity"] = train["Heart_Rate"] / train["Duration"]
test["Intensity"] = test["Heart_Rate"] / test["Duration"]


# Label Encoding
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train["Sex"] = train["Sex"].astype("category")
test["Sex"] = test["Sex"].astype("category")


# Prepare train/test matrices
X = train.drop(columns=["id", "Calories"])
y = np.log1p(train["Calories"])
X_test = test.drop(columns=["id"])
FOLDS = 5

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(train))
pred = np.zeros(len(test))


# Cross-validation
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\nğŸ”� Fold {i+1}")
    
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    start = time.time()
    
    model = XGBRegressor(
        tree_method="hist",  # CPU version
        predictor="auto",
        max_depth=10,
        colsample_bytree=0.7,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=0.01,
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True,
        verbosity=0
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    oof[valid_idx] = model.predict(x_valid)
    pred += model.predict(X_test)

    fold_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof[valid_idx])))
    print(f"ğŸ“‰ Fold {i+1} RMSLE: {fold_rmsle:.5f}")
    print(f"â�±ï¸� Fold time: {time.time() - start:.1f} sec")


# Final RMSLE
pred /= FOLDS
full_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof)))
print(f"\nâœ… Final RMSLE: {full_rmsle:.5f}")


# Predictions and submission
y_preds = np.expm1(pred)
print('Mean prediction:', y_preds.mean())
print('Median prediction:', np.median(y_preds))

y_preds = np.clip(y_preds, 1, 314)
submission["Calories"] = y_preds
submission.to_csv("submission.csv", index=False)
submission.head()

