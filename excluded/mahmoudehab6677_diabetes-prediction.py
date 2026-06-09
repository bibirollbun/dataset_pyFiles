import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

TARGET = "diagnosed_diabetes"
ID_COL = "id"

y = train[TARGET]
X = train.drop([TARGET], axis=1)
X_test = test.copy()



def feature_engineering(df):
    df = df.copy()
    
    # Interactions
    df["bmi_age"] = df["bmi"] * df["age"]
    df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1)
    df["chol_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)
    
    df["lifestyle_risk"] = (
        df["smoking_status"] +
        df["alcohol_consumption_per_week"] +
        df["screen_time_hours_per_day"]
    )
    
    return df

X = feature_engineering(X)
X_test = feature_engineering(X_test)



cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]



def target_encode(train_df, test_df, y, cat_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    global_mean = y.mean()
    
    for col in cat_cols:
        means = train_df.groupby(col)[TARGET].mean()
        train_df[col] = train_df[col].map(means)
        test_df[col] = test_df[col].map(means)
        
        train_df[col].fillna(global_mean, inplace=True)
        test_df[col].fillna(global_mean, inplace=True)
        
    return train_df, test_df



lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.3,
    "verbosity": -1,
    "seed": SEED
}



skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Target Encoding (fold-safe)
    X_tr, X_val = target_encode(X_tr, X_val, y_tr, cat_cols)
    _, X_test_enc = target_encode(X_tr, X_test, y_tr, cat_cols)
    
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_valid = lgb.Dataset(X_val, y_val)
    
    model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        num_boost_round=4000,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(0)
        ]
    )
    
    oof_pred[val_idx] = model.predict(X_val)
    test_pred += model.predict(X_test_enc) / N_SPLITS

print("CV AUC:", roc_auc_score(y, oof_pred))



scaler = StandardScaler()
X_scaled = scaler.fit_transform(oof_pred.reshape(-1, 1))

log_reg = LogisticRegression()
log_reg.fit(X_scaled, y)

test_scaled = scaler.transform(test_pred.reshape(-1, 1))
log_pred = log_reg.predict_proba(test_scaled)[:, 1]

final_pred = 0.85 * test_pred + 0.15 * log_pred



submission = pd.DataFrame({
    "id": test[ID_COL],
    "diabetes": final_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()


