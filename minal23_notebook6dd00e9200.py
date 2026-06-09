# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/ss4gg-hackathon-nir-neospectra'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Kaggle-ready baseline pipeline for SOC prediction (NIR + geo-covariates)
# Save this file as `kaggle_baseline_soc.py` or run within a Kaggle notebook cell.


import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

RANDOM_SEED = 42
N_SPLITS = 5
N_PCA = 50  # number of PCA components for spectra

def find_input_path():
    # Typical Kaggle input path; adjust if your dataset folder name differs
    base = '/kaggle/input'
    for d in os.listdir(base):
        if 'ss4gg' in d.lower() or 'neospectra' in d.lower():
            return os.path.join(base, d)
    return base

INPUT_DIR = find_input_path()
print('Using input dir:', INPUT_DIR)

# filenames expected in the input directory
TRAIN_CSV = os.path.join(INPUT_DIR, 'train.csv')
TRAIN_GEO = os.path.join(INPUT_DIR, 'train_geocovariates.csv')
TEST_CSV = os.path.join(INPUT_DIR, 'test.csv')
TEST_GEO = os.path.join(INPUT_DIR, 'test_geocovariates.csv')
SAMPLE_SUB = os.path.join(INPUT_DIR, 'sample_submission.csv')

# load files
train = pd.read_csv(TRAIN_CSV)
train_geo = pd.read_csv(TRAIN_GEO)
test = pd.read_csv(TEST_CSV)
test_geo = pd.read_csv(TEST_GEO)
sub = pd.read_csv(SAMPLE_SUB)

print('train shape', train.shape)
print('train_geo shape', train_geo.shape)
print('test shape', test.shape)
print('test_geo shape', test_geo.shape)

# Identify spectral columns: numeric names between 1350 and 2550 (as strings)
def get_spectral_cols(df):
    spec = [c for c in df.columns if c.replace('.', '', 1).isdigit()]
    spec = sorted(spec, key=lambda x: float(x))
    return spec

spec_cols = get_spectral_cols(train)
print('Detected spectral bands:', len(spec_cols), 'range', spec_cols[0], 'to', spec_cols[-1])

# Merge geo covariates (left join by sample_id)
train = train.merge(train_geo, on='sample_id', how='left')
test = test.merge(test_geo, on='sample_id', how='left')

# Prepare target and IDs
y = train['soc_perc_log1p'].values
train_ids = train['sample_id'].values
test_ids = test['sample_id'].values

# Build spectral matrix
X_spec_train = train[spec_cols].astype(float).values
X_spec_test = test[spec_cols].astype(float).values

# Basic spectral preprocessing: log transform avoided because reflectance 0-1; just scale
# Apply PCA on spectra
print('Fitting PCA on spectra...')
pca = PCA(n_components=N_PCA, random_state=RANDOM_SEED)
X_spec_train_pca = pca.fit_transform(X_spec_train)
X_spec_test_pca = pca.transform(X_spec_test)
print('Explained variance by PCA (sum):', pca.explained_variance_ratio_.sum())

# Prepare geo features: drop IDs, spectral columns and target
drop_cols = set(['sample_id', 'soc_perc_log1p']) | set(spec_cols)
geo_cols = [c for c in train.columns if c not in drop_cols]
print('Number of geo covariates detected:', len(geo_cols))

X_geo_train = train[geo_cols].copy()
X_geo_test = test[geo_cols].copy()

# Impute and scale geo covariates
imp = SimpleImputer(strategy='median')
X_geo_train_imp = imp.fit_transform(X_geo_train)
X_geo_test_imp = imp.transform(X_geo_test)

scaler = StandardScaler()
X_geo_train_s = scaler.fit_transform(X_geo_train_imp)
X_geo_test_s = scaler.transform(X_geo_test_imp)

# Concatenate PCA spectral features and geo covariates
X_train = np.hstack([X_spec_train_pca, X_geo_train_s])
X_test = np.hstack([X_spec_test_pca, X_geo_test_s])
print('Final feature shapes:', X_train.shape, X_test.shape)

# Free memory
del X_spec_train, X_spec_test, X_spec_train_pca, X_spec_test_pca, X_geo_train, X_geo_test
gc.collect()

# Cross-validated LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': RANDOM_SEED,
    'verbosity': -1,
}

folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
preds_test = np.zeros(X_test.shape[0])
cv_scores = []

for fold, (tr_idx, val_idx) in enumerate(folds.split(X_train, y)):
    print(f'Fold {fold+1}')
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
    params,
    train_data,
    num_boost_round=3000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(100)
    ]
)


    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    print('  RMSE val:', rmse)
    cv_scores.append(rmse)

    preds_test += model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS

print('CV RMSE mean/std:', np.mean(cv_scores), np.std(cv_scores))

# Prepare submission
submission = pd.DataFrame({'sample_id': test_ids, 'soc_perc_log1p': preds_test})
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)
print('Wrote submission to', submission_path)

# Save out-of-fold predictions (optional)
# (If you want OOF predictions, modify the loop to collect them.)

print('\nDone. Tips:')
print('- You can tune N_PCA and LightGBM params to improve performance.')
print('- Try spectral smoothing, derivatives, or continuum removal for better spectral signal.')
print('- Consider stacking models (XGBoost, RandomForest) or neural spectral models (1D-CNN).')



# TOPSIS-based model comparison (Kaggle-ready)
# Save as a single cell in Kaggle and run.
import os, time, pickle, gc
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

RND=42
N_SPLITS=4
N_PCA=50

# --- load data (adjust input dir if needed) ---
INPUT_DIR = '/kaggle/input/ss4gg-hackathon-nir-neospectra'
train = pd.read_csv(os.path.join(INPUT_DIR,'train.csv'))
train_geo = pd.read_csv(os.path.join(INPUT_DIR,'train_geocovariates.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR,'test.csv'))
test_geo = pd.read_csv(os.path.join(INPUT_DIR,'test_geocovariates.csv'))

# spectral columns detection
def spec_cols_of(df):
    s = [c for c in df.columns if c.replace('.','',1).isdigit()]
    return sorted(s, key=lambda x: float(x))
spec_cols = spec_cols_of(train)
print("Spectral bands:", len(spec_cols), spec_cols[0], "to", spec_cols[-1])

# merge geo covariates
train = train.merge(train_geo, on='sample_id', how='left')
test = test.merge(test_geo, on='sample_id', how='left')

y = train['soc_perc_log1p'].values
train_ids = train['sample_id'].values

# Spectra -> PCA
X_spec = train[spec_cols].astype(float).values
pca = PCA(n_components=N_PCA, random_state=RND)
X_spec_pca = pca.fit_transform(X_spec)

# Geo features
drop_set = set(['sample_id','soc_perc_log1p'])|set(spec_cols)
geo_cols = [c for c in train.columns if c not in drop_set]
X_geo = train[geo_cols].copy()
imp = SimpleImputer(strategy='median')
X_geo_imp = imp.fit_transform(X_geo)
scaler = StandardScaler()
X_geo_scaled = scaler.fit_transform(X_geo_imp)

# Final feature matrix
X = np.hstack([X_spec_pca, X_geo_scaled])
print("Feature shape:", X.shape)

# Define models to compare (wrap parameters to be fast on Kaggle)
models = {
    'LightGBM': lambda: lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05,
                                          num_leaves=31, random_state=RND, verbose=-1),
    'XGBoost': lambda: xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, 
                                        max_depth=6, objective='reg:squarederror', random_state=RND, verbosity=0),
    'RandomForest': lambda: RandomForestRegressor(n_estimators=400, max_depth=20, random_state=RND, n_jobs=-1),
    'Ridge': lambda: Ridge(alpha=1.0, random_state=RND),
    'SVR_rbf': lambda: SVR(kernel='rbf', C=1.0, gamma='scale')
}

# storage for results
results = []

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

for name, model_fn in models.items():
    print("\n=== Model:", name)
    rmses, maes, r2s = [], [], []
    train_times, pred_times = [], []
    # We'll serialize one fitted model per fold to measure size; average size used
    model_sizes = []
    fold = 0
    for tr_idx, val_idx in kf.split(X, y):
        fold += 1
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]
        m = model_fn()
        # train timer
        t0 = time.time()
        m.fit(Xtr, ytr)
        train_times.append(time.time()-t0)
        # predict timer
        t0 = time.time()
        ypred = m.predict(Xval)
        pred_times.append((time.time()-t0)/len(val_idx))  # per-sample predict time (sec)
        # metrics
        rmses.append(mean_squared_error(yval, ypred, squared=False))
        maes.append(mean_absolute_error(yval, ypred))
        r2s.append(r2_score(yval, ypred))
        # serialize and measure model size bytes
        buf = pickle.dumps(m)
        model_sizes.append(len(buf))
        # free mem
        del m
        gc.collect()
    # aggregate
    res = {
        'model': name,
        'rmse_mean': np.mean(rmses),
        'rmse_std': np.std(rmses),
        'mae_mean': np.mean(maes),
        'r2_mean': np.mean(r2s),
        'train_time_mean': np.mean(train_times),
        'pred_time_per_sample_mean': np.mean(pred_times),
        'model_size_bytes_mean': np.mean(model_sizes)
    }
    print(res)
    results.append(res)

df_res = pd.DataFrame(results).set_index('model')
print("\nCV summary:\n", df_res.round(4))
df_res.to_csv('cv_model_metrics.csv')

# ----------------------
# TOPSIS ranking
# ----------------------
# Criteria: lower is better for rmse, mae, pred_time, model_size; higher better for r2.
# We'll invert R2 to make it a 'higher is better' column separately in TOPSIS step.
# Build decision matrix
dm = df_res.copy()

# Define criteria directions: 1 = benefit (higher better), -1 = cost (lower better)
criteria = {
    'rmse_mean': -1,
    'mae_mean': -1,
    'r2_mean': 1,
    'pred_time_per_sample_mean': -1,
    'model_size_bytes_mean': -1
}
criteria_list = list(criteria.keys())
W = {k:1.0 for k in criteria_list}  # equal weights by default
# You can change weights. Example: give RMSE double weight:
# W['rmse_mean'] = 2.0

# construct matrix (models x criteria)
mat = dm[criteria_list].values.astype(float)
# Normalize by vector magnitude
norm = np.linalg.norm(mat, axis=0)
norm_mat = mat / norm

# Apply weights
weights = np.array([W[c] for c in criteria_list])
weighted = norm_mat * weights

# Determine ideal best and worst
ideal_best = np.max(weighted * np.array([1 if criteria[c]==1 else -1 for c in criteria_list]), axis=0)
ideal_worst = np.min(weighted * np.array([1 if criteria[c]==1 else -1 for c in criteria_list]), axis=0)

# But simpler standard TOPSIS: for benefit criteria use max, cost use min:
ideal_best = np.array([weighted[:,i].max() if criteria_list[i] in criteria_list and criteria[criteria_list[i]]==1 else weighted[:,i].min() for i in range(len(criteria_list))])
ideal_worst = np.array([weighted[:,i].min() if criteria_list[i] in criteria_list and criteria[criteria_list[i]]==1 else weighted[:,i].max() for i in range(len(criteria_list))])

# distances
dist_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))
# TOPSIS score
score = dist_worst / (dist_best + dist_worst)
df_rank = dm.copy()
df_rank['topsis_score'] = score
df_rank = df_rank.sort_values('topsis_score', ascending=False)
df_rank['rank'] = np.arange(1, len(df_rank)+1)
print("\nTOPSIS ranking:\n", df_rank[['topsis_score','rank']].round(4))
df_rank.to_csv('model_ranking.csv')

# show full table
print("\nFull metrics + TOPSIS:\n", df_rank.round(4))

# Save df_rank
df_rank.to_csv('model_ranking.csv')
print("\nSaved model_ranking.csv and cv_model_metrics.csv")



# Retrain best model (LightGBM) on full training data and generate submission
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Reload data quickly
INPUT_DIR = '/kaggle/input/ss4gg-hackathon-nir-neospectra'
train = pd.read_csv(os.path.join(INPUT_DIR,'train.csv'))
train_geo = pd.read_csv(os.path.join(INPUT_DIR,'train_geocovariates.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR,'test.csv'))
test_geo = pd.read_csv(os.path.join(INPUT_DIR,'test_geocovariates.csv'))

train = train.merge(train_geo, on='sample_id', how='left')
test = test.merge(test_geo, on='sample_id', how='left')

y = train['soc_perc_log1p'].values
spec_cols = [c for c in train.columns if c.replace('.', '', 1).isdigit()]
spec_cols = sorted(spec_cols, key=lambda x: float(x))

# PCA on spectra
N_PCA = 50
pca = PCA(n_components=N_PCA, random_state=42)
X_spec_train = pca.fit_transform(train[spec_cols])
X_spec_test = pca.transform(test[spec_cols])

# Geo covariates
drop_cols = set(['sample_id','soc_perc_log1p']) | set(spec_cols)
geo_cols = [c for c in train.columns if c not in drop_cols]
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_geo_train = scaler.fit_transform(imp.fit_transform(train[geo_cols]))
X_geo_test = scaler.transform(imp.transform(test[geo_cols]))

# Combine
X_train = np.hstack([X_spec_train, X_geo_train])
X_test = np.hstack([X_spec_test, X_geo_test])

# Train final LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'seed': 42,
}
train_data = lgb.Dataset(X_train, label=y)
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.log_evaluation(100)]
)

# Predict on test
preds = model.predict(X_test)

# Create submission
sub = pd.DataFrame({
    'sample_id': test['sample_id'],
    'soc_perc_log1p': preds
})
sub.to_csv('submission.csv', index=False)
print('✅ submission.csv saved — upload this to Kaggle!')



# Retrain LightGBM with CV to get best iteration, then train final model and save files.
import os, gc, pickle, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# ---------- Config ----------
INPUT_DIR = '/kaggle/input/ss4gg-hackathon-nir-neospectra'
N_SPLITS = 4            # use the same K as your CV
RND = 42
N_PCA = 50
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'seed': RND,
    'verbose': -1
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 100
OUT_MODEL_TXT = 'best_lightgb_model.txt'
OUT_MODEL_PICKLE = 'best_lightgb_model.pkl'
OUT_SUB = 'submission.csv'

# ---------- Load ----------
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
train_geo = pd.read_csv(os.path.join(INPUT_DIR, 'train_geocovariates.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
test_geo = pd.read_csv(os.path.join(INPUT_DIR, 'test_geocovariates.csv'))

train = train.merge(train_geo, on='sample_id', how='left')
test = test.merge(test_geo, on='sample_id', how='left')

y = train['soc_perc_log1p'].values
spec_cols = [c for c in train.columns if c.replace('.', '', 1).isdigit()]
spec_cols = sorted(spec_cols, key=lambda x: float(x))

# ---------- Preprocess ----------
# PCA on spectra
pca = PCA(n_components=N_PCA, random_state=RND)
X_spec = pca.fit_transform(train[spec_cols].astype(float).values)
X_spec_test = pca.transform(test[spec_cols].astype(float).values)

# Geo covariates
drop_set = set(['sample_id','soc_perc_log1p']) | set(spec_cols)
geo_cols = [c for c in train.columns if c not in drop_set]
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_geo = scaler.fit_transform(imp.fit_transform(train[geo_cols]))
X_geo_test = scaler.transform(imp.transform(test[geo_cols]))

# Final matrices
X = np.hstack([X_spec, X_geo])
X_test = np.hstack([X_spec_test, X_geo_test])
print("Feature shapes:", X.shape, X_test.shape)

# ---------- CV to find best iteration ----------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)
best_iters = []
oof_preds = np.zeros(X.shape[0])
fold_rmses = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nFold {fold}/{N_SPLITS}")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val)
    bst = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING),
                   lgb.log_evaluation(period=0)]
    )
    best_iter = bst.best_iteration or bst.num_trees()
    print("  Best iteration:", best_iter)
    best_iters.append(best_iter)
    # predict val
    pred_val = bst.predict(X_val, num_iteration=best_iter)
    oof_preds[val_idx] = pred_val
    rmse = mean_squared_error(y_val, pred_val, squared=False)
    fold_rmses.append(rmse)
    print("  Fold RMSE:", rmse)
    # free
    del bst, dtrain, dvalid
    gc.collect()

mean_best_iter = int(np.round(np.mean(best_iters)))
cv_rmse_mean = float(np.round(np.mean(fold_rmses), 6))
cv_rmse_std = float(np.round(np.std(fold_rmses), 6))
print(f"\nCV results: mean_best_iter={mean_best_iter}, RMSE_mean={cv_rmse_mean}, RMSE_std={cv_rmse_std}")

# ---------- Retrain final model on full data ----------
print("\nRetraining final model on full data with num_boost_round =", mean_best_iter)
dfull = lgb.Dataset(X, label=y)
final_bst = lgb.train(
    LGB_PARAMS,
    dfull,
    num_boost_round=mean_best_iter,
    callbacks=[lgb.log_evaluation(period=100)]
)

# ---------- Save model ----------
print("Saving model to", OUT_MODEL_TXT, "and pickled object", OUT_MODEL_PICKLE)
final_bst.save_model(OUT_MODEL_TXT, num_iteration=mean_best_iter)
# Also save a pickle wrapper (useful to reload in python)
with open(OUT_MODEL_PICKLE, 'wb') as f:
    pickle.dump(final_bst, f)

# ---------- Predict test and save submission ----------
print("Predicting test set...")
preds_test = final_bst.predict(X_test, num_iteration=mean_best_iter)
submission = pd.DataFrame({'sample_id': test['sample_id'], 'soc_perc_log1p': preds_test})
submission.to_csv(OUT_SUB, index=False)
print("Saved submission to", OUT_SUB)

# ---------- Summary ----------
print("\nDONE. Files created in working dir:")
print(" -", OUT_MODEL_TXT)
print(" -", OUT_MODEL_PICKLE)
print(" -", OUT_SUB)
print(f"\nFinal CV RMSE (mean ± std): {cv_rmse_mean} ± {cv_rmse_std}")

# If you want to download these files locally from Kaggle UI, use the file browser on the right.



# Fixed ensemble script: XGBoost + LightGBM (avoids DMatrix <-> lgb mixups and object dtype issues)
import os, gc, pickle, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

# Config
INPUT_DIR = '/kaggle/input/ss4gg-hackathon-nir-neospectra'
OUT_SUB = 'submission_ensemble.csv'
OUT_LGB = 'best_lightgb_model.txt'
OUT_XGB = 'best_xgb_model.bin'
N_SPLITS = 4
RND = 42
N_PCA = 50
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'seed': RND,
    'verbose': -1
}
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'seed': RND,
    'verbosity': 0
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 100

# Load
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
train_geo = pd.read_csv(os.path.join(INPUT_DIR, 'train_geocovariates.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
test_geo = pd.read_csv(os.path.join(INPUT_DIR, 'test_geocovariates.csv'))

train = train.merge(train_geo, on='sample_id', how='left')
test = test.merge(test_geo, on='sample_id', how='left')

y = train['soc_perc_log1p'].values
spec_cols = [c for c in train.columns if c.replace('.', '', 1).isdigit()]
spec_cols = sorted(spec_cols, key=lambda x: float(x))

# Preprocess: PCA on spectra + impute/scale geo
pca = PCA(n_components=N_PCA, random_state=RND)
X_spec_train = pca.fit_transform(train[spec_cols].astype(float).values)
X_spec_test = pca.transform(test[spec_cols].astype(float).values)

drop_set = set(['sample_id','soc_perc_log1p']) | set(spec_cols)
geo_cols = [c for c in train.columns if c not in drop_set]
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_geo_train = scaler.fit_transform(imp.fit_transform(train[geo_cols]))
X_geo_test = scaler.transform(imp.transform(test[geo_cols]))

# Final feature matrices — FORCE numeric dtype (float32)
X = np.hstack([X_spec_train, X_geo_train]).astype(np.float32)
X_test = np.hstack([X_spec_test, X_geo_test]).astype(np.float32)
y = y.astype(np.float32)

print("Feature shapes:", X.shape, X_test.shape)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

# --- XGBoost CV & final model ---
oof_xgb = np.zeros(X.shape[0], dtype=np.float32)
test_preds_xgb = np.zeros(X_test.shape[0], dtype=np.float32)
xgb_best_iters = []
xgb_fold_rmse = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nXGB Fold {fold}/{N_SPLITS}")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    bst_xgb = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False
    )
    # use best_iteration if available (xgboost returns 0-based best_iteration)
    best_iter = getattr(bst_xgb, 'best_iteration', None)
    if best_iter is None:
        best_iter = NUM_BOOST_ROUND
    else:
        best_iter = int(best_iter) + 1
    xgb_best_iters.append(best_iter)
    # predict using iteration_range (works on modern xgboost)
    oof_xgb[val_idx] = bst_xgb.predict(xgb.DMatrix(X_val), iteration_range=(0, best_iter)).astype(np.float32)
    test_preds_xgb += bst_xgb.predict(xgb.DMatrix(X_test), iteration_range=(0, best_iter)).astype(np.float32) / N_SPLITS
    rmse = mean_squared_error(y_val, oof_xgb[val_idx], squared=False)
    xgb_fold_rmse.append(rmse)
    print("  best_iter:", best_iter, "fold RMSE:", rmse)
    del bst_xgb, dtrain, dvalid
    gc.collect()

xgb_mean_iter = int(np.round(np.mean([it for it in xgb_best_iters if it is not None])))
print(f"\nXGBoost CV mean_iter={xgb_mean_iter}, RMSE_mean={np.mean(xgb_fold_rmse):.6f} ± {np.std(xgb_fold_rmse):.6f}")

# Save full XGBoost on full train
dtrain_full = xgb.DMatrix(X, label=y)
final_xgb = xgb.train(XGB_PARAMS, dtrain_full, num_boost_round=xgb_mean_iter, verbose_eval=False)
final_xgb.save_model(OUT_XGB)
print("Saved XGBoost model to", OUT_XGB)

# --- LightGBM: if model exists, load and compute OOF by re-training per-fold with same best_iter ---
lgb_oof = np.zeros(X.shape[0], dtype=np.float32)
test_preds_lgb = np.zeros(X_test.shape[0], dtype=np.float32)
lgb_best_iters = []

if os.path.exists(OUT_LGB):
    print("\nFound existing LightGBM:", OUT_LGB, "- will compute CV folds by retraining per-fold using appropriate early stopping.")
    # We'll do folds to get OOF and averaged test predictions
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\nLGB Fold {fold}/{N_SPLITS}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)
        bst_lgb = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dtr, dval],
            valid_names=['train','valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING), lgb.log_evaluation(period=0)]
        )
        best_iter = bst_lgb.best_iteration or bst_lgb.num_trees()
        lgb_best_iters.append(best_iter)
        lgb_oof[val_idx] = bst_lgb.predict(X_val, num_iteration=best_iter).astype(np.float32)
        test_preds_lgb += bst_lgb.predict(X_test, num_iteration=best_iter).astype(np.float32) / N_SPLITS
        print("  Fold best_iter:", best_iter, "fold RMSE:", mean_squared_error(y_val, lgb_oof[val_idx], squared=False))
        del bst_lgb, dtr, dval
        gc.collect()
    lgb_mean_iter = int(np.round(np.mean(lgb_best_iters)))
    print("LightGBM mean_iter from folds:", lgb_mean_iter)
else:
    print("\nNo LightGBM model found; performing LGB CV then training final LGB.")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\nLGB Fold {fold}/{N_SPLITS}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)
        bst_lgb = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dtr, dval],
            valid_names=['train','valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING), lgb.log_evaluation(period=0)]
        )
        best_iter = bst_lgb.best_iteration or bst_lgb.num_trees()
        lgb_best_iters.append(best_iter)
        lgb_oof[val_idx] = bst_lgb.predict(X_val, num_iteration=best_iter).astype(np.float32)
        test_preds_lgb += bst_lgb.predict(X_test, num_iteration=best_iter).astype(np.float32) / N_SPLITS
        print("  Fold best_iter:", best_iter, "fold RMSE:", mean_squared_error(y_val, lgb_oof[val_idx], squared=False))
        del bst_lgb, dtr, dval
        gc.collect()
    lgb_mean_iter = int(np.round(np.mean(lgb_best_iters)))
    print("\nRetraining final LightGBM on full data with num_boost_round =", lgb_mean_iter)
    dfull = lgb.Dataset(X, label=y)
    final_lgb = lgb.train(LGB_PARAMS, dfull, num_boost_round=lgb_mean_iter, callbacks=[lgb.log_evaluation(period=100)])
    final_lgb.save_model(OUT_LGB, num_iteration=lgb_mean_iter)
    print("Saved LightGBM model to", OUT_LGB)
    # After retrain, for final test use predictions below (we already computed averaged test_preds_lgb during CV)

# --- Ensemble OOF evaluation ---
rmse_xgb = mean_squared_error(y, oof_xgb, squared=False)
rmse_lgb = mean_squared_error(y, lgb_oof, squared=False)
oof_ensemble = (oof_xgb + lgb_oof) / 2.0
rmse_ens = mean_squared_error(y, oof_ensemble, squared=False)
print(f"\nOOF RMSEs -> XGB: {rmse_xgb:.6f}, LGB: {rmse_lgb:.6f}, Ensemble(avg): {rmse_ens:.6f}")

# --- Ensemble test predictions and save ---
test_preds = (test_preds_xgb + test_preds_lgb) / 2.0
submission = pd.DataFrame({'sample_id': test['sample_id'], 'soc_perc_log1p': test_preds})
submission.to_csv(OUT_SUB, index=False)
print("Saved ensemble submission to", OUT_SUB)

# Save final_xgb as pickle too
with open('final_xgb.pkl', 'wb') as f:
    pickle.dump(final_xgb, f)
print("Saved final_xgb.pkl")



# Full recompute + blending: trains XGBoost & LightGBM (CV), computes OOF & test preds,
# then creates weighted avg and stacking submissions and picks the best by OOF RMSE.
# Saves: submission_weighted.csv, submission_stacked.csv, submission_best.csv
import os, gc, pickle, numpy as np, pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

# ---------- Config ----------
INPUT_DIR = '/kaggle/input/ss4gg-hackathon-nir-neospectra'
OUT_SUB_WEIGHTED = 'submission_weighted.csv'
OUT_SUB_STACKED = 'submission_stacked.csv'
OUT_SUB_BEST = 'submission_best.csv'
OUT_LGB = 'best_lightgb_model.txt'
OUT_XGB = 'best_xgb_model.bin'

N_SPLITS = 4
RND = 42
N_PCA = 50
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 100

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'seed': RND,
    'verbose': -1
}
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'seed': RND,
    'verbosity': 0
}

# ---------- Load ----------
train = pd.read_csv(os.path.join(INPUT_DIR,'train.csv'))
train_geo = pd.read_csv(os.path.join(INPUT_DIR,'train_geocovariates.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR,'test.csv'))
test_geo = pd.read_csv(os.path.join(INPUT_DIR,'test_geocovariates.csv'))

train = train.merge(train_geo, on='sample_id', how='left')
test = test.merge(test_geo, on='sample_id', how='left')

y = train['soc_perc_log1p'].values.astype(np.float32)
spec_cols = [c for c in train.columns if c.replace('.', '', 1).isdigit()]
spec_cols = sorted(spec_cols, key=lambda x: float(x))

# ---------- Preprocess ----------
pca = PCA(n_components=N_PCA, random_state=RND)
X_spec_train = pca.fit_transform(train[spec_cols].astype(float).values)
X_spec_test = pca.transform(test[spec_cols].astype(float).values)

drop_set = set(['sample_id','soc_perc_log1p']) | set(spec_cols)
geo_cols = [c for c in train.columns if c not in drop_set]
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_geo_train = scaler.fit_transform(imp.fit_transform(train[geo_cols]))
X_geo_test = scaler.transform(imp.transform(test[geo_cols]))

X = np.hstack([X_spec_train, X_geo_train]).astype(np.float32)
X_test = np.hstack([X_spec_test, X_geo_test]).astype(np.float32)
print("Feature shapes:", X.shape, X_test.shape)

# ---------- CV folds ----------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

# ---------- XGBoost CV ----------
oof_xgb = np.zeros(X.shape[0], dtype=np.float32)
test_preds_xgb = np.zeros(X_test.shape[0], dtype=np.float32)
xgb_best_iters = []
xgb_fold_rmse = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nXGB Fold {fold}/{N_SPLITS}")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    bst = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain,'train'), (dvalid,'valid')],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False
    )
    best_iter = getattr(bst, 'best_iteration', None)
    if best_iter is None:
        best_iter = NUM_BOOST_ROUND
    else:
        best_iter = int(best_iter) + 1
    xgb_best_iters.append(best_iter)
    oof_xgb[val_idx] = bst.predict(xgb.DMatrix(X_val), iteration_range=(0, best_iter)).astype(np.float32)
    test_preds_xgb += bst.predict(xgb.DMatrix(X_test), iteration_range=(0, best_iter)).astype(np.float32) / N_SPLITS
    rmse = mean_squared_error(y_val, oof_xgb[val_idx], squared=False)
    xgb_fold_rmse.append(rmse)
    print("  best_iter:", best_iter, "fold RMSE:", rmse)
    del bst, dtrain, dvalid
    gc.collect()

xgb_mean_iter = int(np.round(np.mean([it for it in xgb_best_iters if it is not None])))
print(f"\nXGB CV mean_iter={xgb_mean_iter}, RMSE_mean={np.mean(xgb_fold_rmse):.6f} ± {np.std(xgb_fold_rmse):.6f}")

# save final xgb on full data
dtrain_full = xgb.DMatrix(X, label=y)
final_xgb = xgb.train(XGB_PARAMS, dtrain_full, num_boost_round=xgb_mean_iter, verbose_eval=False)
final_xgb.save_model(OUT_XGB)
print("Saved XGB model to", OUT_XGB)

# ---------- LightGBM CV ----------
oof_lgb = np.zeros(X.shape[0], dtype=np.float32)
test_preds_lgb = np.zeros(X_test.shape[0], dtype=np.float32)
lgb_best_iters = []
lgb_fold_rmse = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nLGB Fold {fold}/{N_SPLITS}")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val)
    bst = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dvalid],
        valid_names=['train','valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING), lgb.log_evaluation(period=0)]
    )
    best_iter = bst.best_iteration or bst.num_trees()
    lgb_best_iters.append(best_iter)
    oof_lgb[val_idx] = bst.predict(X_val, num_iteration=best_iter).astype(np.float32)
    test_preds_lgb += bst.predict(X_test, num_iteration=best_iter).astype(np.float32) / N_SPLITS
    rmse = mean_squared_error(y_val, oof_lgb[val_idx], squared=False)
    lgb_fold_rmse.append(rmse)
    print("  best_iter:", best_iter, "fold RMSE:", rmse)
    del bst, dtrain, dvalid
    gc.collect()

lgb_mean_iter = int(np.round(np.mean(lgb_best_iters)))
print(f"\nLGB CV mean_iter={lgb_mean_iter}, RMSE_mean={np.mean(lgb_fold_rmse):.6f} ± {np.std(lgb_fold_rmse):.6f}")

# save final lgb on full data
dfull = lgb.Dataset(X, label=y)
final_lgb = lgb.train(LGB_PARAMS, dfull, num_boost_round=lgb_mean_iter, callbacks=[lgb.log_evaluation(period=100)])
final_lgb.save_model(OUT_LGB)
print("Saved LGB model to", OUT_LGB)

# ---------- Blending: weighted & stacking ----------
rmse_xgb = mean_squared_error(y, oof_xgb, squared=False)
rmse_lgb = mean_squared_error(y, oof_lgb, squared=False)
print("\nBase OOF RMSEs -> XGB:", rmse_xgb, "LGB:", rmse_lgb)

# Weighted average (weights inverse to RMSE)
w_x = (1.0 / rmse_xgb)
w_l = (1.0 / rmse_lgb)
w_sum = w_x + w_l
w_x /= w_sum
w_l /= w_sum
print("Weights -> XGB: {:.4f}, LGB: {:.4f}".format(w_x, w_l))

weighted_oof = w_x * oof_xgb + w_l * oof_lgb
rmse_weighted = mean_squared_error(y, weighted_oof, squared=False)
print("Weighted OOF RMSE:", rmse_weighted)

test_preds_weighted = w_x * test_preds_xgb + w_l * test_preds_lgb
pd.DataFrame({'sample_id': test['sample_id'], 'soc_perc_log1p': test_preds_weighted}).to_csv(OUT_SUB_WEIGHTED, index=False)
print("Saved", OUT_SUB_WEIGHTED)

# Stacking meta-learner (Ridge) with CV for meta OOF estimation
X_meta = np.vstack([oof_xgb, oof_lgb]).T
X_meta_test = np.vstack([test_preds_xgb, test_preds_lgb]).T

meta = Ridge(alpha=1.0)
meta_oof = cross_val_predict(meta, X_meta, y, cv=kf, method='predict')
rmse_meta_cv = mean_squared_error(y, meta_oof, squared=False)
print("Meta (Ridge) CV OOF RMSE:", rmse_meta_cv)

meta.fit(X_meta, y)
meta_test_preds = meta.predict(X_meta_test)
pd.DataFrame({'sample_id': test['sample_id'], 'soc_perc_log1p': meta_test_preds}).to_csv(OUT_SUB_STACKED, index=False)
print("Saved", OUT_SUB_STACKED)

# Choose best by OOF RMSE
if rmse_meta_cv < rmse_weighted:
    chosen = 'stacked'
    chosen_preds = meta_test_preds
    chosen_rmse = rmse_meta_cv
    chosen_file = OUT_SUB_STACKED
else:
    chosen = 'weighted'
    chosen_preds = test_preds_weighted
    chosen_rmse = rmse_weighted
    chosen_file = OUT_SUB_WEIGHTED

pd.DataFrame({'sample_id': test['sample_id'], 'soc_perc_log1p': chosen_preds}).to_csv(OUT_SUB_BEST, index=False)
print("\nChosen:", chosen, " (OOF RMSE = {:.6f}). Saved as".format(chosen_rmse), OUT_SUB_BEST)
print("\nAll done. Files in working directory:", OUT_SUB_WEIGHTED, OUT_SUB_STACKED, OUT_SUB_BEST)


