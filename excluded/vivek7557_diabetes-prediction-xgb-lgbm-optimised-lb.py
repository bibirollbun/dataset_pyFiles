import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeClassifier
from sklearn.isotonic import IsotonicRegression
from scipy.stats import rankdata
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 7
seeds = [42, 52, 62]
TARGET = 'diagnosed_diabetes'

# ===============================
# LOAD DATA
# ===============================
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

y = train[TARGET]
X = train.drop(columns=[TARGET])
X_test = test.copy()

# ===============================
# FIX CATEGORICAL ISSUES (IMPORTANT)
# ===============================
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    
    X[col].fillna('Unknown', inplace=True)
    X_test[col].fillna('Unknown', inplace=True)
    
    le = LabelEncoder()
    full = pd.concat([X[col], X_test[col]], axis=0)
    le.fit(full)
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])

# ===============================
# CV TRAINING
# ===============================
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

for seed in seeds:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=48,
            max_depth=9,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=seed
        )
        
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='auc'
        )
        
        oof_lgb[va] += lgb_model.predict_proba(X_va)[:, 1] / len(seeds)
        test_lgb += lgb_model.predict_proba(X_test)[:, 1] / (N_FOLDS * len(seeds))
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='auc',
            random_state=seed
        )
        
        xgb_model.fit(X_tr, y_tr)
        
        oof_xgb[va] += xgb_model.predict_proba(X_va)[:, 1] / len(seeds)
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / (N_FOLDS * len(seeds))

# ===============================
# BLEND OPTIMIZATION
# ===============================

def blend_auc(w):
    p = w[0]*oof_lgb + w[1]*oof_xgb
    return -roc_auc_score(y, p)

res = minimize(blend_auc, [0.5, 0.5], bounds=((0,1),(0,1)))
w_lgb, w_xgb = res.x

blend_oof = w_lgb*oof_lgb + w_xgb*oof_xgb
print('OOF AUC:', roc_auc_score(y, blend_oof))

blend_test = w_lgb*test_lgb + w_xgb*test_xgb

# ===============================
# CALIBRATION
# ===============================
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(blend_oof, y)
cal_test = iso.transform(blend_test)

# ===============================
# RANK AVERAGING (PRIVATE LB SAFE)
# ===============================
final_pred = rankdata(cal_test) / len(cal_test)
final_pred = np.clip(final_pred, 0.002, 0.998)

# ===============================
# SUBMISSION
# ===============================
submission = pd.DataFrame({
    'id': sample['id'],
    TARGET: final_pred
})

submission.to_csv('submission_private_lb.csv', index=False)
print('Saved submission_private_lb.csv')





