# Import libraries
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# Remove duplicates & reduce synthetic leakage
cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
train = train.drop_duplicates(subset=train.columns, keep='first').reset_index(drop=True)
train = train.groupby(by=cols)['Calories'].min().reset_index()


# Encode 'Sex'
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])  # male=1, female=0
test['Sex'] = le.transform(test['Sex'])


# Feature Engineering
def add_bmi_intensity(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    return df

def add_cross_terms(df, features):
    df = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            name = f"{features[i]}_x_{features[j]}"
            df[name] = df[features[i]] * df[features[j]]
    return df

numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
for df in [train, test]:
    df = add_bmi_intensity(df)
    df = add_cross_terms(df, numerical_features)


# Prepare matrices
X = train.drop(columns=["Calories"])
y = np.log1p(train["Calories"])
X_test = test.drop(columns=["id"])


# Stratified KFold based on Duration bins
n_bins = 10
duration_binned = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)\
                    .fit_transform(train[["Duration"]]).astype(int).flatten()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Ensemble CV
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_binned)):
    print(f"\\nğŸ”� Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    start = time.time()

    # --- CatBoost ---
    cat = CatBoostRegressor(
        verbose=0,
        random_state=42
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    oof_preds[val_idx] += cat.predict(X_val) * 0.5
    cat_preds += cat.predict(X_test) / skf.n_splits

    # --- XGBoost ---
    xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.7,
        gamma=0.01,
        max_delta_step=2,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        eval_metric="rmse",
        verbosity=0,
        random_state=42
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    oof_preds[val_idx] += xgb.predict(X_val) * 0.5
    xgb_preds += xgb.predict(X_test) / skf.n_splits

    fold_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(oof_preds[val_idx])))
    print(f"ğŸ“‰ Fold {fold+1} RMSLE: {fold_rmsle:.5f}")
    print(f"â�±ï¸� Time: {time.time() - start:.1f} sec")


# Final Evaluation
final_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
print(f"\\nâœ… Final OOF RMSLE: {final_rmsle:.5f}")


# Final submission
final_preds = 0.5 * cat_preds + 0.5 * xgb_preds
submission["Calories"] = np.clip(np.expm1(final_preds), 1, 314)
submission.to_csv("submission.csv", index=False)
print("\\nğŸ“� submission.csv saved.")
submission.head()

