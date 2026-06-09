import numpy as np
import pandas as pd
import os
import gc
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")
SEED = 42


TRAIN_PATH = "/kaggle/input/playground-series-s5e10/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e10/test.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

# Keep ids for submission
test_ids = test['id'].copy()


target_col = "accident_risk"
assert target_col in train.columns, "Train must contain target column"


train['is_train'] = 1
test['is_train']  = 0
test[target_col] = np.nan
df = pd.concat([train, test], ignore_index=True)


for c in df.select_dtypes("object").columns:
    df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan":np.nan})

# Convert booleans / 0-1 like columns to ints if they're not numeric
bool_like = ['public_road','school_season','holiday']
for col in bool_like:
    if col in df.columns:
        df[col] = df[col].map({True:1, False:0, 'true':1, 'false':0}).astype(float)


df['curvature_speed_ratio'] = df['curvature'] / (df['speed_limit'].fillna(0) + 1)
df['lanes_speed_product']   = df['num_lanes'].fillna(0) * df['speed_limit'].fillna(0)
df['curvature_lane_interaction'] = df['curvature'].fillna(0) * df['num_lanes'].fillna(0)

df['is_high_speed']    = (df['speed_limit'] >= 60).astype(int)
df['is_high_curvature']= (df['curvature'] > df['curvature'].median()).astype(int)

# lighting flags (accepts many text forms)
df['is_low_light'] = df['lighting'].fillna('').str.contains('dim|dark|low', case=False, regex=True).astype(int)
df['is_night']     = df['time_of_day'].fillna('').str.contains('night|evening', case=False, regex=True).astype(int)
df['is_peak_hour'] = df['time_of_day'].fillna('').str.contains('morning|afternoon|evening', case=False, regex=True).astype(int)

# weather flags
df['is_rainy'] = df['weather'].fillna('').str.contains('rain|shower|storm', case=False, regex=True).astype(int)
df['is_foggy'] = df['weather'].fillna('').str.contains('fog|mist', case=False, regex=True).astype(int)
df['is_clear'] = df['weather'].fillna('').str.contains('clear|sun', case=False, regex=True).astype(int)
df['bad_weather_flag'] = ((df['is_rainy']==1) | (df['is_foggy']==1)).astype(int) * df['is_low_light']

# accidents stats
df['num_reported_accidents'] = df['num_reported_accidents'].fillna(0)
df['log_num_reported_accidents'] = np.log1p(df['num_reported_accidents'])
df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'].fillna(0) + 1)
df['accidents_per_speed'] = df['num_reported_accidents'] / (df['speed_limit'].fillna(0) + 1)

# composite
df['risky_condition_score'] = (
    df['is_high_speed'].fillna(0) +
    df['is_high_curvature'].fillna(0) +
    df['is_low_light'].fillna(0) +
    df['is_rainy'].fillna(0) +
    df['is_foggy'].fillna(0) +
    (~df['road_signs_present'].astype(bool)).astype(int).fillna(0)
)
df['risk_factor_index'] = df['risky_condition_score'] / (df['risky_condition_score'].max() + 1)

# numeric interaction examples
df['speed_per_lane'] = df['speed_limit'].fillna(0) / (df['num_lanes'].fillna(1))
df['curvature_sq'] = df['curvature'].fillna(0) ** 2



cat_cols = [c for c in ['road_type','lighting','weather','time_of_day'] if c in df.columns]
num_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c not in [target_col, 'is_train', 'id']]

# keep any engineered features that are numeric
engineered = [c for c in df.columns if c not in (cat_cols + num_cols + [target_col, 'is_train', 'id'])]
# but we already added engineered as numeric in many cases; rebuild num_cols to include them
num_cols = sorted(list(set(num_cols + engineered)))

print("Categorical columns:", cat_cols)
print("Numeric columns (sample):", num_cols[:10])


train = df[df['is_train']==1].reset_index(drop=True)
test  = df[df['is_train']==0].reset_index(drop=True)

X = train[num_cols + cat_cols].copy()
y = train[target_col].astype(float).copy()
X_test = test[num_cols + cat_cols].copy()



class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols, cat_cols, seed=SEED):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seed = seed
        self.num_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.target_encoders = {c: TargetEncoder(cols=[c]) for c in cat_cols}

    def fit(self, X, y=None):
        if len(self.num_cols)>0:
            self.num_imputer.fit(X[self.num_cols])
            self.scaler.fit(self.num_imputer.transform(X[self.num_cols]))
        # fit target encoders with y if available (important)
        if y is not None:
            for c, enc in self.target_encoders.items():
                enc.fit(X[c].astype(str), y)
        else:
            # fit on X only to avoid errors (fallback)
            for c, enc in self.target_encoders.items():
                enc.fit(X[c].astype(str), np.zeros(len(X)))
        return self

    def transform(self, X):
        X_ = X.copy()
        # numeric
        if len(self.num_cols)>0:
            X_num = pd.DataFrame(self.num_imputer.transform(X_[self.num_cols]), columns=self.num_cols, index=X_.index)
            X_num = pd.DataFrame(self.scaler.transform(X_num), columns=self.num_cols, index=X_.index)
        else:
            X_num = pd.DataFrame(index=X_.index)
        # categorical -> target encode
        X_cat = pd.DataFrame(index=X_.index)
        for c, enc in self.target_encoders.items():
            # enc.transform expects DataFrame-like; category_encoders TargetEncoder returns numpy array-like
            X_cat[c+'_te'] = enc.transform(X_[c].astype(str))
        out = pd.concat([X_num, X_cat], axis=1)
        return out


# Paste this cell before running Step 9 (or replace Step 9 with this)
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import xgboost as xgb

NFOLDS = 5
SEED = 42

# ---------- Ensure X, y, X_test exist ----------
# If you already have X (train features), y (target), and X_test (test features), keep using them.
# If not, try to derive them from train/test DataFrames in the notebook:
if 'X' not in globals() or 'y' not in globals() or 'X_test' not in globals():
    # attempt to derive from train/test variables if they exist
    if 'train' in globals() and 'test' in globals() and 'target_col' in globals():
        X = train.drop(columns=[target_col, 'is_train', 'id'], errors='ignore').select_dtypes(include=[np.number, object]).copy()
        y = train[target_col].astype(float).copy()
        X_test = test.drop(columns=[target_col, 'is_train', 'id'], errors='ignore').select_dtypes(include=[np.number, object]).copy()
    else:
        raise RuntimeError("Please make sure variables X, y, X_test or train/test with target_col are defined before running this cell.")

# ---------- If preproc exists, use it; otherwise fallback to simple preprocessing ----------
if 'preproc' in globals():
    # use the existing preprocessor (target encoders + imputer + scaler) if available
    try:
        X_tr = preproc.transform(X)
        X_te = preproc.transform(X_test)
    except Exception as e:
        print("preproc exists but failed to transform, falling back to simple preprocessing:", e)
        preproc = None
else:
    preproc = None

if preproc is None:
    # Build a simple robust preprocessing pipeline (safe fallback)
    # Identify categorical columns (object or category types)
    cat_cols_local = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype).startswith('category')]
    num_cols_local = [c for c in X.columns if c not in cat_cols_local]

    # Fill missing values for numeric and scale
    num_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_num = pd.DataFrame(num_imputer.fit_transform(X[num_cols_local]), columns=num_cols_local, index=X.index)
    X_num = pd.DataFrame(scaler.fit_transform(X_num), columns=num_cols_local, index=X.index)

    # For test
    X_test_num = pd.DataFrame(num_imputer.transform(X_test[num_cols_local]), columns=num_cols_local, index=X_test.index)
    X_test_num = pd.DataFrame(scaler.transform(X_test_num), columns=num_cols_local, index=X_test.index)

    # Simple label encoding for categorical columns (deterministic)
    X_cat = pd.DataFrame(index=X.index)
    X_test_cat = pd.DataFrame(index=X_test.index)
    for c in cat_cols_local:
        # fit on combined to avoid unseen labels
        combined = pd.concat([X[c].astype(str), X_test[c].astype(str)], axis=0)
        labels, uniques = pd.factorize(combined, sort=True)
        X_cat[c+'_le'] = labels[:len(X)]
        X_test_cat[c+'_le'] = labels[len(X):]

    # Final matrices
    X_tr = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    X_te = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)
    # ensure same columns order
    X_tr.columns = [str(c) for c in X_tr.columns]
    X_te.columns = [str(c) for c in X_te.columns]
    # Align columns (add missing)
    missing_cols = set(X_tr.columns) - set(X_te.columns)
    for c in missing_cols:
        X_te[c] = 0
    extra_cols = set(X_te.columns) - set(X_tr.columns)
    for c in extra_cols:
        X_te.drop(columns=[c], inplace=True)

# ---------- Sanity: ensure indexes are integers and aligned ----------
X_tr = X_tr.reset_index(drop=True)
X_te = X_te.reset_index(drop=True)
y = y.reset_index(drop=True)

print("Prepared shapes -> X_tr:", X_tr.shape, "X_te:", X_te.shape, "y:", y.shape)

# ---------- Initialize OOF/Test arrays (original names used) ----------
oof_preds_lgb = np.zeros(len(X_tr))
test_preds_lgb = np.zeros(len(X_te))
oof_preds_xgb = np.zeros(len(X_tr))
test_preds_xgb = np.zeros(len(X_te))

# ---------- Model params ----------
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': SEED
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'alpha': 0.1,
    'verbosity': 0,
    'seed': SEED
}

kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

# ---------- Training loop (keeps your original variable names) ----------
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr, y)):
    print(f"\n==== Fold {fold+1}/{NFOLDS} ====")
    X_train_fold, X_val_fold = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[tr_idx], y.iloc[val_idx]

    # LightGBM dataset
    dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
    dvalid = lgb.Dataset(X_val_fold, label=y_val_fold, reference=dtrain)

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )

    oof_preds_lgb[val_idx] = lgb_model.predict(X_val_fold, num_iteration=lgb_model.best_iteration)
    test_preds_lgb += lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration) / NFOLDS

    # XGBoost
    dtrain_x = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval_x   = xgb.DMatrix(X_val_fold, label=y_val_fold)
    watchlist = [(dtrain_x, 'train'), (dval_x, 'valid')]

    xgb_model = xgb.train(
        xgb_params,
        dtrain_x,
        num_boost_round=5000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=200
    )

    best_iter = xgb_model.best_iteration if hasattr(xgb_model, "best_iteration") else 0

    # use modern prediction API (iteration_range)
    try:
        oof_preds_xgb[val_idx] = xgb_model.predict(dval_x, iteration_range=(0, best_iter))
        test_preds_xgb += xgb_model.predict(xgb.DMatrix(X_te), iteration_range=(0, best_iter)) / NFOLDS
    except TypeError:
        # fallback for older xgboost versions
        oof_preds_xgb[val_idx] = xgb_model.predict(dval_x, ntree_limit=getattr(xgb_model, "best_ntree_limit", None))
        test_preds_xgb += xgb_model.predict(xgb.DMatrix(X_te), ntree_limit=getattr(xgb_model, "best_ntree_limit", None)) / NFOLDS

    # Cleanup
    del dtrain, dvalid, dtrain_x, dval_x, lgb_model, xgb_model
    gc.collect()

print("\n✅ Training complete across all folds!")



oof_avg = (oof_preds_lgb + oof_preds_xgb) / 2
test_avg = (test_preds_lgb + test_preds_xgb) / 2

rmse_lgb = mean_squared_error(y, oof_preds_lgb, squared=False)
rmse_xgb = mean_squared_error(y, oof_preds_xgb, squared=False)
rmse_avg = mean_squared_error(y, oof_avg, squared=False)

print("\nCV RMSE LightGBM: {:.6f}".format(rmse_lgb))
print("CV RMSE XGBoost : {:.6f}".format(rmse_xgb))
print("CV RMSE Average : {:.6f}".format(rmse_avg))

from sklearn.linear_model import Ridge
meta_X = np.vstack([oof_preds_lgb, oof_preds_xgb]).T
meta_test_X = np.vstack([test_preds_lgb, test_preds_xgb]).T

meta = Ridge(alpha=1.0, random_state=SEED)
meta.fit(meta_X, y)
oof_meta = meta.predict(meta_X)
test_meta = meta.predict(meta_test_X)
rmse_meta = mean_squared_error(y, oof_meta, squared=False)
print("CV RMSE Stacked (Ridge) : {:.6f}".format(rmse_meta))

# Choose final predictions (we use stacked)
final_test_preds = test_meta



submission = pd.DataFrame({'id': test_ids, target_col: final_test_preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Saved submission to /kaggle/working/submission.csv")

# Quick feature importance from the last LightGBM (if still in memory)
try:
    # If lgb_model exists in scope, else skip
    importances = None
except Exception:
    pass

