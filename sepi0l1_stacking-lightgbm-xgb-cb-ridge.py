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


import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error



def rmsle(y_true, y_pred):
    return mean_squared_log_error(y_true, np.maximum(y_pred, 0)) ** 0.5

def feature_engineering(df):
    df = df.copy()

    # BMI
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

    # EtkileÅŸimler
    df['Duration_x_HeartRate'] = df['Duration'] * df['Heart_Rate']
    df['Duration_x_Temp'] = df['Duration'] * df['Body_Temp']
    df['HeartRate_x_Temp'] = df['Heart_Rate'] * df['Body_Temp']
    df['BMI_x_Duration'] = df['BMI'] * df['Duration']
    df['BMI_x_HeartRate'] = df['BMI'] * df['Heart_Rate']

    # Oranlar
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['HeartRate_per_age'] = df['Heart_Rate'] / df['Age']
    df['Duration_per_age'] = df['Duration'] / df['Age']
    df['HeartRate_per_kg'] = df['Heart_Rate'] / df['Weight']
    df['Temp_per_age'] = df['Body_Temp'] / df['Age']
    df['BMI_per_age'] = df['BMI'] / df['Age']

    # Polinomlar
    df['Duration_sq'] = df['Duration'] ** 2
    df['HeartRate_sq'] = df['Heart_Rate'] ** 2
    df['Temp_sq'] = df['Body_Temp'] ** 2
    df['Duration_cube'] = df['Duration'] ** 3

    # Log-transformlar
    df['Log_Duration'] = np.log1p(df['Duration'])
    df['Log_HeartRate'] = np.log1p(df['Heart_Rate'])
    df['Log_BMI'] = np.log1p(df['BMI'])
    df['Weight_log'] = np.log1p(df['Weight'])

    # Log etkileÅŸimler
    df["log_DurHR"] = np.log1p(df["Duration"] * df["Heart_Rate"])
    df["log_DurBMI"] = np.log1p(df["Duration"] * df["BMI"])
    df["log_HRTemp"] = np.log1p(df["Heart_Rate"] * df["Body_Temp"])

    # KÃ¶k
    df["HeartRate_root"] = np.sqrt(df["Heart_Rate"])

    # Binning
    df['Age_bin'] = pd.qcut(df['Age'], q=4, labels=False)
    df['Duration_bin'] = pd.qcut(df['Duration'], q=4, labels=False)

    return df




    return df



train.columns


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train["Sex"] = le.fit_transform(train["Sex"])
test["Sex"] = le.transform(test["Sex"])

train = feature_engineering(train)
test = feature_engineering(test)

X = train.drop(columns=["id", "Calories"])
y = train["Calories"]
X_test = test.drop(columns=["id"])



from lightgbm import early_stopping, log_evaluation

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

# log dÃ¶nÃ¼ÅŸÃ¼m uygulanmÄ±ÅŸ y
y_log = np.log1p(y)

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_log = y_log.iloc[train_idx]
    y_val_log = y_log.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        learning_rate=0.0634, max_depth=8, num_leaves=122, min_child_samples=49,
        subsample=0.7763, colsample_bytree=0.8529, reg_alpha=0.8080, reg_lambda=0.3676,
        n_estimators=500, random_state=42
    )
    lgb_model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric="rmse",
        callbacks=[
            early_stopping(50),
            log_evaluation(100)
        ]
    )
    oof_lgb[val_idx] = np.expm1(lgb_model.predict(X_val))
    test_preds_lgb += np.expm1(lgb_model.predict(X_test)) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor( 
        learning_rate=0.0531, max_depth=10, min_child_weight=7, gamma=0.7593,
        subsample=0.8541, colsample_bytree=0.7027, reg_alpha=1.0995, reg_lambda=1.0626,
        n_estimators=500, tree_method="hist", random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], early_stopping_rounds=50, verbose=False)
    oof_xgb[val_idx] = np.expm1(xgb_model.predict(X_val))
    test_preds_xgb += np.expm1(xgb_model.predict(X_test)) / kf.n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        learning_rate=0.1911, depth=10, l2_leaf_reg=1.0819,
        bagging_temperature=0.1315, random_strength=0.0530,
        iterations=500, verbose=0, random_state=42,
        early_stopping_rounds=50
    )
    cat_model.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), cat_features=["Sex", "Age_bin"])
    oof_cat[val_idx] = np.expm1(cat_model.predict(X_val))
    test_preds_cat += np.expm1(cat_model.predict(X_test)) / kf.n_splits



from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Stack input: base modellerin tahminleri zaten normal space'te (expm1 ile dÃ¶nmÃ¼ÅŸ)
X_level2 = np.column_stack((oof_lgb, oof_xgb, oof_cat))
X_test_level2 = np.column_stack((test_preds_lgb, test_preds_xgb, test_preds_cat))

# Ridge modeli orijinal hedefle eÄŸit (log deÄŸil!)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_level2, y)

# OOF tahminler
oof_meta = ridge_model.predict(X_level2)
oof_meta = np.maximum(oof_meta, 0)

# RMSLE
rmsle_score = mean_squared_log_error(y, oof_meta) ** 0.5
print(f"âœ… Ridge Meta Model RMSLE: {rmsle_score:.5f}")

# Final test tahmini
final_test_preds = ridge_model.predict(X_test_level2)
final_test_preds = np.maximum(final_test_preds, 0)



train


from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Meta input ve target
X_level2 = np.column_stack((oof_lgb, oof_xgb, oof_cat))

# Ridge meta model
ridge = Ridge(alpha=1.0)
ridge.fit(X_level2, y   )
oof_ridge = ridge.predict(X_level2)

# LGBM meta model
lgb_meta = lgb.LGBMRegressor(
    learning_rate=0.0634,
    max_depth=8,
    num_leaves=122,
    min_child_samples=49,
    subsample=0.7763,
    colsample_bytree=0.8529,
    reg_alpha=0.8080,
    reg_lambda=0.3676,
    n_estimators=500,
    random_state=42
)
lgb_meta.fit(X_level2, y)
oof_lgbm_meta = lgb_meta.predict(X_level2)



train


best_score = float("inf")
best_weight = None

for w in np.linspace(0, 1, 21):  # 0.00, 0.05, ..., 1.00
    blended = w * oof_ridge + (1 - w) * oof_lgbm_meta
    blended = np.maximum(blended, 0)
    score = mean_squared_log_error(y, blended) ** 0.5
    print(f"Weight {w:.2f} â†’ RMSLE: {score:.5f}")
    if score < best_score:
        best_score = score
        best_weight = w

print(f"\nâœ… En iyi aÄŸÄ±rlÄ±k: Ridge {best_weight:.2f} | LGBM {1-best_weight:.2f}")
print(f"ğŸ“‰ En dÃ¼ÅŸÃ¼k RMSLE: {best_score:.5f}")



X_test_level2 = np.column_stack((test_preds_lgb, test_preds_xgb, test_preds_cat))

# Ridge test tahminleri (meta_preds_ridge)
meta_preds_ridge = ridge.predict(X_test_level2)
meta_preds_ridge = np.maximum(meta_preds_ridge, 0)

# LGBM test tahminleri (meta_preds_lgbm)
meta_preds_lgbm = lgb_meta.predict(X_test_level2)
meta_preds_lgbm = np.maximum(meta_preds_lgbm, 0)



from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Stack Level 2 preds
X_level3 = np.column_stack((oof_ridge, oof_lgbm_meta))
X_test_level3 = np.column_stack((meta_preds_ridge, meta_preds_lgbm))

# Ridge meta model
ridge_lvl3 = Ridge(alpha=1.0)
ridge_lvl3.fit(X_level3, y)

# CV OOF prediction
oof_lvl3 = ridge_lvl3.predict(X_level3)
oof_lvl3 = np.maximum(oof_lvl3, 0)
rmsle_lvl3 = mean_squared_log_error(y, oof_lvl3) ** 0.5
print(f"ğŸ“Š Level 3 Ridge RMSLE: {rmsle_lvl3:.5f}")

# Test prediction
final_preds_lvl3 = ridge_lvl3.predict(X_test_level3)
final_preds_lvl3 = np.maximum(final_preds_lvl3, 0)

# Submission
submission = pd.DataFrame({
    "id": test["id"],
    "Calories": final_preds_lvl3
})
submission.to_csv("submission.csv", index=False)
print("ğŸš€ Level 3 Ridge submission.csv created.")





