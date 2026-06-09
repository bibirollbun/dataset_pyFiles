# Setup & imports
import os, gc, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
SEED = 42
np.random.seed(SEED)

# try important libs
try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    has_lgb = False
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    has_xgb = False
try:
    import catboost as cb
    has_cat = True
except Exception:
    has_cat = False

print('has_lgb', has_lgb, 'has_xgb', has_xgb, 'has_cat', has_cat)


# Paths (Kaggle competition)
TRAIN_PATH = '/kaggle/input/playground-series-s5e10/train.csv'
TEST_PATH  = '/kaggle/input/playground-series-s5e10/test.csv'
SAMPLE_PATH = '/kaggle/input/playground-series-s5e10/sample_submission.csv'

assert os.path.exists(TRAIN_PATH), "Train file not found at expected path."
assert os.path.exists(TEST_PATH), "Test file not found at expected path."

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)
print('train', train.shape, 'test', test.shape)
train.head()


# Quick checks
print(train.dtypes)
print('\nTarget distribution:')
if 'accident_risk' in train.columns:
    print(train['accident_risk'].describe())
else:
    print('Target column not found: expected "accident_risk".')


# Copy and basic conversions
df = train.copy()
test_df = test.copy()

# Ensure booleans -> ints
for c in df.columns:
    if df[c].dtype == 'bool':
        df[c] = df[c].astype(int)
        test_df[c] = test_df[c].astype(int)

# Basic features (adapted from original notebook)
df['high_speed'] = (df['speed_limit'] >= 60).astype(int)
df['curvature_sq'] = df['curvature']**2
df['curvature_sqrt'] = (df['curvature'].abs() + 1e-9)**0.5
df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'].replace(0,1))
df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'].replace(0,1))
df['public_no_signs'] = ((df['public_road'] == 1) & (df['road_signs_present'] == False)).astype(int)

# same for test
test_df['high_speed'] = (test_df['speed_limit'] >= 60).astype(int)
test_df['curvature_sq'] = test_df['curvature']**2
test_df['curvature_sqrt'] = (test_df['curvature'].abs() + 1e-9)**0.5
test_df['speed_per_lane'] = test_df['speed_limit'] / (test_df['num_lanes'].replace(0,1))
test_df['accidents_per_lane'] = test_df['num_reported_accidents'] / (test_df['num_lanes'].replace(0,1))
test_df['public_no_signs'] = ((test_df['public_road'] == 1) & (test_df['road_signs_present'] == False)).astype(int)

# Identify features
TARGET = 'accident_risk'
ID_COL = 'id'
features = [c for c in df.columns if c not in [TARGET, ID_COL]]
# Move target to y
y = df[TARGET].values
print('Feature count:', len(features))


# ðŸ”§ Encode categorical features
cat_cols = [c for c in df.columns if df[c].dtype == 'object']
print("Categorical columns:", cat_cols)

for c in cat_cols:
    df[c], uniques = pd.factorize(df[c])
    test_df[c] = pd.Categorical(test_df[c], categories=uniques).codes

# Prepare matrices
X = df[features].values
X_test = test_df[features].values

NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof = { 'lgb': np.zeros(len(df)), 'xgb': np.zeros(len(df)), 'cat': np.zeros(len(df)), 'hgb': np.zeros(len(df)) }
preds = { 'lgb': np.zeros(len(test_df)), 'xgb': np.zeros(len(test_df)), 'cat': np.zeros(len(test_df)), 'hgb': np.zeros(len(test_df)) }

def train_lgb(X_tr, y_tr, X_val, y_val):
    params = {
        'objective':'regression','metric':'rmse','learning_rate':0.03,'num_leaves':128,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l1':0.3,'lambda_l2':0.6,
        'verbosity':-1,'seed':SEED
    }
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
    params, dtr, num_boost_round=20000, valid_sets=[dtr, dval],
    callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)]
)

    return model

def train_xgb(X_tr, y_tr, X_val, y_val):
    params = {'objective':'reg:squarederror','learning_rate':0.03,'max_depth':8,
              'min_child_weight':3,'subsample':0.9,'colsample_bytree':0.9,
              'reg_lambda':1.2,'reg_alpha':0.3,'n_estimators':5000,'tree_method':'hist','random_state':SEED}
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], early_stopping_rounds=300, verbose=False)
    return model

def train_cat(X_tr, y_tr, X_val, y_val):
    params = {'iterations':3000,'learning_rate':0.03,'depth':8,'l2_leaf_reg':4,'random_seed':SEED,'loss_function':'RMSE','verbose':False}
    model = cb.CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val,y_val), early_stopping_rounds=300, verbose=False)
    return model

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print('Fold', fold+1)
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # LGB
    if has_lgb:
        try:
            m = train_lgb(X_tr, y_tr, X_val, y_val)
            oof['lgb'][val_idx] = m.predict(X_val, num_iteration=m.best_iteration)
            preds['lgb'] += m.predict(X_test, num_iteration=m.best_iteration) / NFOLDS
            print(' lgb ok')
        except Exception as e:
            print(' lgb fail', e)

    # XGB
    if has_xgb:
        try:
            m = train_xgb(X_tr, y_tr, X_val, y_val)
            oof['xgb'][val_idx] = m.predict(X_val)
            preds['xgb'] += m.predict(X_test) / NFOLDS
            print(' xgb ok')
        except Exception as e:
            print(' xgb fail', e)

    # Cat
    if has_cat:
        try:
            m = train_cat(X_tr, y_tr, X_val, y_val)
            oof['cat'][val_idx] = m.predict(X_val)
            preds['cat'] += m.predict(X_test) / NFOLDS
            print(' cat ok')
        except Exception as e:
            print(' cat fail', e)

    # HGB
    try:
        h = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, random_state=SEED)
        h.fit(X_tr, y_tr)
        oof['hgb'][val_idx] = h.predict(X_val)
        preds['hgb'] += h.predict(X_test) / NFOLDS
        print(' hgb ok')
    except Exception as e:
        print(' hgb fail', e)

# OOF scores
for k,v in oof.items():
    if v.sum() != 0:
        print(k, 'OOF RMSE:', mean_squared_error(y, v, squared=False))



# Build meta features
meta_X = np.column_stack([oof['lgb'], oof['xgb'], oof['cat'], oof['hgb']])
meta_test = np.column_stack([preds['lgb'], preds['xgb'], preds['cat'], preds['hgb']])

# Ridge meta-learner
meta = Ridge(alpha=1.0, random_state=SEED)
meta.fit(meta_X, y)
meta_oof = meta.predict(meta_X)
print('Meta OOF RMSE (Ridge):', mean_squared_error(y, meta_oof, squared=False))

# Convex blend optimization
preds_arr = meta_X  # using OOFs
x0 = np.array([0.25,0.25,0.25,0.25])
cons = ({'type':'eq','fun': lambda w: 1 - np.sum(w)})
bounds = [(0,1)]*preds_arr.shape[1]
def blend_loss(w):
    w = np.array(w)
    w = np.maximum(w,0)
    if w.sum()==0:
        w = np.ones_like(w)
    w = w / w.sum()
    blended = np.dot(preds_arr, w)
    return mean_squared_error(y, blended, squared=False)

res = minimize(blend_loss, x0, method='SLSQP', bounds=bounds, constraints=cons)
weights = res.x
weights = np.maximum(weights,0); weights = weights / weights.sum()
print('Optimized weights (lgb,xgb,cat,hgb):', weights)

blended_oof = np.dot(preds_arr, weights)
print('Blended OOF RMSE:', mean_squared_error(y, blended_oof, squared=False))


# Residual modeling with small XGB on training features
try:
    res_model = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=0)
    res_model.fit(X, (y - blended_oof))
    print('Residual model trained.')
    use_res = True
except Exception as e:
    print('Residual model unavailable:', e)
    use_res = False

# (Optional) pseudo-labeling - disabled by default
DO_PSEUDO = False
if DO_PSEUDO and use_res:
    test_stack_arr = meta_test
    blended_test = np.dot(test_stack_arr, weights)
    if use_res:
        blended_test += res_model.predict(X_test)
    low_q, high_q = np.quantile(blended_test, [0.01,0.99])
    mask_conf = (blended_test <= low_q) | (blended_test >= high_q)
    print('Pseudo candidates:', mask_conf.sum())


# Final test blend
test_blended = np.dot(meta_test, weights)
if use_res:
    try:
        test_blended += res_model.predict(X_test)
    except Exception:
        pass
test_blended = np.clip(test_blended, 0, 1)

submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: test_blended})
submission.to_csv('submission.csv', index=False)
print('Saved submission.csv â€” rows:', len(submission))
submission.head()

