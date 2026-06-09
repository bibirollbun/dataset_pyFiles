import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Reproducibility
SEED = 42
np.random.seed(SEED)


DATA_DIR = '/kaggle/input/playground-series-s5e5/'
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")


# Generating all pairwise cross-terms
num_cols = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
for i, f1 in enumerate(num_cols):
    for f2 in num_cols[i+1:]:
        train_df[f"{f1}_x_{f2}"] = train_df[f1] * train_df[f2]
        test_df [f"{f1}_x_{f2}"] = test_df[f1] * test_df[f2]


# Rank by absolute correlation with target
cross_feats = [c for c in train_df if '_x_' in c]
corrs = train_df[cross_feats].corrwith(train_df['Calories']).abs()
top_feats = corrs.sort_values(ascending=False).head(10).index.tolist()
print("Top 10 interaction features:", top_feats)

# Reduce dataset to only those
keep = ['id','Sex','Calories'] + num_cols + top_feats
train_df = train_df[keep].copy()
test_df  = test_df [ [c for c in keep if c!='Calories'] ].copy()


le = LabelEncoder()
train_df["Sex_bin"] = le.fit_transform(train_df["Sex"])
test_df["Sex_bin"] = le.transform(test_df["Sex"])


# Features for modeling
FEATURES = num_cols + top_feats + ['Sex_bin']

X = train_df[FEATURES]
y_log = np.log1p(train_df['Calories'])  # for RMSLE-aligned training
X_test = test_df[FEATURES]

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# Allocating arrays for OOF and test predictions
oof_xgb = np.zeros(len(X))
pred_xgb = np.zeros(len(X_test))

oof_lgb = np.zeros(len(X))
pred_lgb = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(kf.split(X),1):
    print(f"--- Fold {fold} ---")

    # Spliting train/valid sets
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y_log.iloc[tr], y_log.iloc[va]

    # XGBoost on log(target)
    xgb = XGBRegressor(
        objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
        learning_rate=0.02, max_depth=10, subsample=0.9, colsample_bytree=0.7,
        gamma=0.01, n_estimators=2000, early_stopping_rounds=100,
        random_state=SEED, verbosity=0
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va,y_va)], verbose=100)

    # OOF & test preds
    p_va = np.expm1(xgb.predict(X_va)).clip(0)
    oof_xgb[va] = p_va
    pred_xgb += np.expm1(xgb.predict(X_test)).clip(0) / FOLDS

    # LightGBM on log(target)
    lgb = LGBMRegressor(
        objective='regression', metric='rmse', learning_rate=0.02, num_leaves=31,
        subsample=0.9, colsample_bytree=0.7, n_estimators=2000,
        random_state=SEED, n_jobs=-1
    )
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va,y_va)],
        eval_metric='rmse',
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)]
    )
    p_va = np.expm1(lgb.predict(X_va)).clip(0)
    oof_lgb[va] = p_va
    pred_lgb += np.expm1(lgb.predict(X_test)).clip(0) / FOLDS

    # 7.3 Fold RMSLEs
    print("XGB RMSLE:", rmsle(train_df['Calories'].iloc[va], oof_xgb[va]))
    print("LGB RMSLE:", rmsle(train_df['Calories'].iloc[va], oof_lgb[va]))


# Computing overall RMSLE
r_xgb = rmsle(train_df['Calories'], oof_xgb)
r_lgb = rmsle(train_df['Calories'], oof_lgb)

# Inverse-error weights
w_xgb = (1/r_xgb) / (1/r_xgb + 1/r_lgb)
w_lgb = 1 - w_xgb
print(f"Weights: XGB: {w_xgb:.2f}, LGB: {w_lgb:.2f}")

blend_pred = w_xgb * pred_xgb + w_lgb * pred_lgb
print("Blended RMSLE:", rmsle(train_df['Calories'],
                                 w_xgb*oof_xgb + w_lgb*oof_lgb))

# Saving blended submission
submission['Calories'] = blend_pred
submission.to_csv('submission_blend.csv', index=False)


# Building stacking train/test frames on original scale
data_stack = pd.DataFrame({'xgb': oof_xgb, 'lgb': oof_lgb})
stack_test = pd.DataFrame({'xgb': pred_xgb, 'lgb': pred_lgb})

y_true = train_df['Calories']

# Training Ridge on true target
meta = Ridge(alpha=1.0, random_state=SEED)
meta.fit(data_stack, y_true)
print("Meta-coeffs:", dict(zip(data_stack.columns, meta.coef_)))

# Meta-predictions
stack_pred = meta.predict(stack_test).clip(0)

# Saving stacked submission
submission['Calories'] = stack_pred
submission.to_csv('submission_stack.csv', index=False)

