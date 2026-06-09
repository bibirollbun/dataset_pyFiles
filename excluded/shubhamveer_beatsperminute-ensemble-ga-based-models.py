import os
import random
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# --- Utility functions ---

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def has_cuda_gpu():
    return torch.cuda.is_available()

GPU = has_cuda_gpu()
print("GPU available:", GPU)

def ensure_cuda(tensor):
    return tensor.cuda() if GPU else tensor

# --- Paths / Config ---

DATA_DIR = '/kaggle/input/playground-series-s5e9/'
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

TARGET_COL = "BeatsPerMinute"
ID_COL = "id"

N_FOLDS = 5
# For hyperopt, use cheaper settings
HP_SEARCH_FOLDS = 3
HP_SEARCH_NROUNDS_SCALE = 0.25  # scale down nrounds during HP

# Autoencoder params
AE_EPOCHS = 30
AE_BATCH = 1024
LATENT_MIN = 8

# MLP final model (if used)
MLP_EPOCHS = 40

# --- Load data ---

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

y = train[TARGET_COL].values
X_raw = train.drop([TARGET_COL, ID_COL], axis=1).reset_index(drop=True)
X_test_raw = test.drop([ID_COL], axis=1).reset_index(drop=True)

print("Loaded train:", train.shape, "test:", test.shape)

# --- Feature engineering: row-stats ---

def add_row_stats(df):
    out = df.copy()
    out['row_mean'] = df.mean(axis=1)
    out['row_std'] = df.std(axis=1)
    out['row_min'] = df.min(axis=1)
    out['row_max'] = df.max(axis=1)
    out['row_skew'] = df.skew(axis=1)
    return out

print("Generating row-level stats...")
X_feats = add_row_stats(X_raw)
X_test_feats = add_row_stats(X_test_raw)

# --- Polynomial features + PCA for dimension reduction ---

print("Creating polynomial features (degree=2), then reducing dimensions via VarThreshold + PCA...")
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

combined = pd.concat([X_feats, X_test_feats], axis=0).reset_index(drop=True)
combined_vals = combined.values

combined_poly = poly.fit_transform(combined_vals)
print("Polynomial feature size before reduction:", combined_poly.shape)

# Remove near-constant features
vt = VarianceThreshold(threshold=1e-6)
combined_poly_reduced = vt.fit_transform(combined_poly)
print("After VarianceThreshold:", combined_poly_reduced.shape)

# PCA to retain most variance
pca = PCA(n_components=0.995, svd_solver='full')  # retain 99.5% variance
combined_pca = pca.fit_transform(combined_poly_reduced)
print("After PCA:", combined_pca.shape)

n_train = X_feats.shape[0]
X_poly = combined_pca[:n_train, :]
X_test_poly = combined_pca[n_train:, :]

# Scale for AE (and other models)
scaler_poly = StandardScaler()
X_poly_scaled = scaler_poly.fit_transform(X_poly)
X_test_poly_scaled = scaler_poly.transform(X_test_poly)

# --- Autoencoder for embeddings ---

print("Training autoencoder embeddings...")
input_dim = X_poly_scaled.shape[1]
latent_dim = max(LATENT_MIN, input_dim // 10)

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, latent):
        super().__init__()
        h = max(128, in_dim // 2)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, latent)  # linear output
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, h),
            nn.ReLU(),
            nn.Linear(h, in_dim)
        )
    def forward(self, x):
        z = self.enc(x)
        xrec = self.dec(z)
        return xrec, z

device = torch.device("cuda" if GPU else "cpu")
ae = AutoEncoder(input_dim, latent_dim).to(device)
ae_opt = optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
ae_loss_fn = nn.MSELoss()

Xae = torch.tensor(X_poly_scaled, dtype=torch.float32).to(device)
n = Xae.shape[0]
for epoch in range(1, AE_EPOCHS + 1):
    perm = np.random.permutation(n)
    epoch_loss = 0.0
    ae.train()
    for i in range(0, n, AE_BATCH):
        idx = perm[i : i + AE_BATCH]
        xb = Xae[idx]
        ae_opt.zero_grad()
        xr, _ = ae(xb)
        loss = ae_loss_fn(xr, xb)
        loss.backward()
        ae_opt.step()
        epoch_loss += float(loss) * xb.size(0)
    epoch_loss /= n
    if epoch % 10 == 0 or epoch == 1:
        print(f"AE Epoch {epoch}/{AE_EPOCHS} loss: {epoch_loss:.6f}")

ae.eval()
with torch.no_grad():
    _, z_train = ae(Xae)
    z_train = z_train.cpu().numpy()
    z_test = ae(torch.tensor(X_test_poly_scaled, dtype=torch.float32).to(device))[1].cpu().numpy()

print("Embeddings shapes:", z_train.shape, z_test.shape)

# --- Final feature matrices ---

print("Building final feature matrices...")
X_final = np.hstack([X_poly_scaled, z_train])
X_test_final = np.hstack([X_test_poly_scaled, z_test])

scaler_final = StandardScaler()
X_final = scaler_final.fit_transform(X_final)
X_test_final = scaler_final.transform(X_test_final)

print("Final feature shapes, train:", X_final.shape, "test:", X_test_final.shape)

# --- OOF / helper for models ---

kf_full = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def get_oof_preds(model_builder, fit_params=None, predict_func=None):
    oof = np.zeros(len(X_final))
    test_preds = np.zeros(X_test_final.shape[0])
    for fold, (tr_idx, val_idx) in enumerate(kf_full.split(X_final), start=1):
        Xtr, Xval = X_final[tr_idx], X_final[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]
        model = model_builder()
        if fit_params is None:
            model.fit(Xtr, ytr)
        else:
            model.fit(Xtr, ytr, **fit_params)
        if predict_func is None:
            oof[val_idx] = model.predict(Xval)
            test_preds += model.predict(X_test_final) / N_FOLDS
        else:
            oof[val_idx] = predict_func(model, Xval)
            test_preds += predict_func(model, X_test_final) / N_FOLDS
        print(f"Fold {fold}/{N_FOLDS} RMSE: {rmse(yval, oof[val_idx]):.6f}")
    return oof, test_preds

# --- Hyperparameter optimization with reduced budget ---

def ga_optimize_hp(eval_fn, param_space, n_gen=6, pop_size=8):
    """
    Genetic Algorithm over param_space.
    param_space: dict name -> (low, high, 'int' or 'float')
    eval_fn: function(params) -> score (lower is better)
    """
    def sample():
        p = {}
        for k, (low, high, ptype) in param_space.items():
            if ptype == 'int':
                p[k] = int(np.random.randint(low, high + 1))
            else:
                p[k] = float(np.random.uniform(low, high))
        return p

    pop = [sample() for _ in range(pop_size)]
    scores = [eval_fn(p) for p in pop]
    for gen in range(1, n_gen + 1):
        ranked = np.argsort(scores)
        topk_idx = ranked[: max(2, pop_size // 2)]
        survivors = [pop[i] for i in topk_idx]
        children = []
        while len(children) < (pop_size - len(survivors)):
            a, b = random.choice(survivors), random.choice(survivors)
            child = {}
            for key in a.keys():
                # crossover
                val = (a[key] + b[key]) / 2.0
                # mutation
                if random.random() < 0.3:
                    span = param_space[key][1] - param_space[key][0]
                    val += np.random.normal(0, 0.08 * span)
                low, high, ptype = param_space[key]
                val = max(low, min(high, val))
                if ptype == 'int':
                    child[key] = int(round(val))
                else:
                    child[key] = float(val)
            children.append(child)
        pop = survivors + children
        scores = [eval_fn(p) for p in pop]
        print(f"GA HP gen {gen}/{n_gen} best so far: {min(scores):.6f}")
    best_idx = int(np.argmin(scores))
    return pop[best_idx], scores[best_idx]

# --- HP search & training for each model type ---

# 1) XGBoost

print("\n=== HP Search for XGBoost ===")

def eval_xgb(params):
    # scale down rounds & folds
    # extract
    max_depth = int(params['max_depth'])
    eta = float(params['eta'])
    subsample = float(params['subsample'])
    colsample = float(params['colsample_bytree'])
    nrounds = int(params['nrounds'])
    # scaled budget
    scaled_nrounds = max(50, int(nrounds * HP_SEARCH_NROUNDS_SCALE))

    xgb_params = {
        'max_depth': max_depth,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample,
        'objective': 'reg:squarederror',
        'verbosity': 0,
        'seed': SEED,
        'tree_method': 'gpu_hist' if GPU else 'hist',
    }

    oof_local = np.zeros(len(X_final))
    kf_local = KFold(n_splits=HP_SEARCH_FOLDS, shuffle=True, random_state=SEED)
    for tr, val in kf_local.split(X_final):
        dtr = xgb.DMatrix(X_final[tr], label=y[tr])
        dval = xgb.DMatrix(X_final[val], label=y[val])
        bst = xgb.train(xgb_params, dtr, num_boost_round=scaled_nrounds,
                        evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=False)
        oof_local[val] = bst.predict(dval)
    return rmse(y, oof_local)

xgb_space = {
    'max_depth': (3, 30, 'int'),
    'eta': (0.01, 0.6, 'float'),
    'subsample': (0.5, 1.0, 'float'),
    'colsample_bytree': (0.5, 1.0, 'float'),
    'nrounds': (100, 800, 'int')
}

best_xgb_params, best_xgb_score = ga_optimize_hp(eval_xgb, xgb_space, n_gen=6, pop_size=10)
print("Best XGB params:", best_xgb_params, "approx RMSE (on reduced CV):", best_xgb_score)

def get_oof_test_xgb(params):
    oof = np.zeros(len(X_final))
    test_preds = np.zeros(X_test_final.shape[0])
    nrounds = int(params['nrounds'])
    xgb_params = {
        'max_depth': int(params['max_depth']),
        'eta': float(params['eta']),
        'subsample': float(params['subsample']),
        'colsample_bytree': float(params['colsample_bytree']),
        'objective': 'reg:squarederror',
        'verbosity': 0,
        'seed': SEED,
        'tree_method': 'gpu_hist' if GPU else 'hist',
    }
    for fold, (tr, val) in enumerate(kf_full.split(X_final), start=1):
        dtr = xgb.DMatrix(X_final[tr], label=y[tr])
        dval = xgb.DMatrix(X_final[val], label=y[val])
        bst = xgb.train(xgb_params, dtr, num_boost_round=nrounds,
                        evals=[(dval, 'val')])
        oof[val] = bst.predict(dval)
        test_preds += bst.predict(xgb.DMatrix(X_test_final)) / N_FOLDS
        print(f"XGB fold {fold} RMSE: {rmse(y[val], oof[val]):.6f}")
    return oof, test_preds

oof_xgb, test_xgb = get_oof_test_xgb(best_xgb_params)

# 2) LightGBM

print("\n=== HP Search for LightGBM ===")

def eval_lgb(params):
    num_leaves = int(params['num_leaves'])
    lr = float(params['learning_rate'])
    feat_frac = float(params['feature_fraction'])
    bag_frac = float(params['bagging_fraction'])
    max_depth = int(params['max_depth'])
    nrounds = int(params['nrounds'])
    scaled_nrounds = max(50, int(nrounds * HP_SEARCH_NROUNDS_SCALE))

    lparams = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': num_leaves,
        'learning_rate': lr,
        'feature_fraction': feat_frac,
        'bagging_fraction': bag_frac,
        'max_depth': max_depth,
        'verbose': -1
    }
    if GPU:
        lparams['device'] = 'gpu'

    oof_local = np.zeros(len(X_final))
    kf_local = KFold(n_splits=HP_SEARCH_FOLDS, shuffle=True, random_state=SEED)
    for tr, val in kf_local.split(X_final):
        dtr = lgb.Dataset(X_final[tr], label=y[tr])
        dval = lgb.Dataset(X_final[val], label=y[val])
        bst = lgb.train(lparams, dtr, num_boost_round=scaled_nrounds,
                        valid_sets=[dval])
        oof_local[val] = bst.predict(X_final[val])
    return rmse(y, oof_local)

lgb_space = {
    'num_leaves': (24, 128, 'int'),
    'learning_rate': (0.01, 0.2, 'float'),
    'feature_fraction': (0.5, 1.0, 'float'),
    'bagging_fraction': (0.5, 1.0, 'float'),
    'max_depth': (3, 12, 'int'),
    'nrounds': (100, 800, 'int')
}

best_lgb_params, best_lgb_score = ga_optimize_hp(eval_lgb, lgb_space, n_gen=6, pop_size=10)
print("Best LGB params:", best_lgb_params, "approx RMSE (on reduced CV):", best_lgb_score)

def get_oof_test_lgb(params):
    oof = np.zeros(len(X_final))
    test_preds = np.zeros(X_test_final.shape[0])
    nrounds = int(params['nrounds'])
    lparams = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': int(params['num_leaves']),
        'learning_rate': float(params['learning_rate']),
        'feature_fraction': float(params['feature_fraction']),
        'bagging_fraction': float(params['bagging_fraction']),
        'max_depth': int(params['max_depth']),
        'verbose': -1
    }
    if GPU:
        lparams['device'] = 'gpu'

    for fold, (tr, val) in enumerate(kf_full.split(X_final), start=1):
        dtr = lgb.Dataset(X_final[tr], label=y[tr])
        dval = lgb.Dataset(X_final[val], label=y[val])
        bst = lgb.train(lparams, dtr, num_boost_round=nrounds,
                        valid_sets=[dval])
        oof[val] = bst.predict(X_final[val])
        test_preds += bst.predict(X_test_final) / N_FOLDS
        print(f"LGB fold {fold} RMSE: {rmse(y[val], oof[val]):.6f}")
    return oof, test_preds

oof_lgb, test_lgb = get_oof_test_lgb(best_lgb_params)

# 3) CatBoost

print("\n=== HP Search for CatBoost ===")

def eval_cb(params):
    depth = int(params['depth'])
    lr = float(params['learning_rate'])
    l2_reg = int(params.get('l2_leaf_reg', 3))
    nrounds = int(params.get('nrounds', 300))
    scaled_nrounds = max(50, int(nrounds * HP_SEARCH_NROUNDS_SCALE))

    oof_local = np.zeros(len(X_final))
    kf_local = KFold(n_splits=HP_SEARCH_FOLDS, shuffle=True, random_state=SEED)
    for tr, val in kf_local.split(X_final):
        model = CatBoostRegressor(
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2_reg,
            iterations=scaled_nrounds,
            task_type='GPU' if GPU else 'CPU',
            verbose=0,
            random_seed=SEED
        )
        model.fit(X_final[tr], y[tr], eval_set=(X_final[val], y[val]),
                  early_stopping_rounds=20, verbose=False)
        oof_local[val] = model.predict(X_final[val])
    return rmse(y, oof_local)

cb_space = {
    'depth': (4, 10, 'int'),
    'learning_rate': (0.01, 0.2, 'float'),
    'l2_leaf_reg': (1, 8, 'int'),
    'nrounds': (100, 600, 'int')
}

best_cb_params, best_cb_score = ga_optimize_hp(eval_cb, cb_space, n_gen=5, pop_size=8)
print("Best CatBoost params:", best_cb_params, "approx RMSE:", best_cb_score)

def get_oof_test_cb(params):
    oof = np.zeros(len(X_final))
    test_preds = np.zeros(X_test_final.shape[0])
    nrounds = int(params.get('nrounds', 300))
    for fold, (tr, val) in enumerate(kf_full.split(X_final), start=1):
        model = CatBoostRegressor(
            depth=int(params['depth']),
            learning_rate=float(params['learning_rate']),
            l2_leaf_reg=int(params.get('l2_leaf_reg', 3)),
            iterations=nrounds,
            task_type='GPU' if GPU else 'CPU',
            verbose=0,
            random_seed=SEED
        )
        model.fit(X_final[tr], y[tr], eval_set=(X_final[val], y[val]),
                  early_stopping_rounds=30, verbose=False)
        oof[val] = model.predict(X_final[val])
        test_preds += model.predict(X_test_final) / N_FOLDS
        print(f"CatBoost fold {fold} RMSE: {rmse(y[val], oof[val]):.6f}")
    return oof, test_preds

oof_cb, test_cb = get_oof_test_cb(best_cb_params)

# Optional: you could also train an MLP similarly if you want, but skipping here for brevity

# --- Stacking / blending via Ridge meta-learner ---

print("\nStacking base model OOFs & evaluating:")

oof_stack = np.vstack([oof_xgb, oof_lgb, oof_cb]).T
test_stack = np.vstack([test_xgb, test_lgb, test_cb]).T

print("Base model RMSEs:")
print("XGB:", rmse(y, oof_xgb))
print("LGB:", rmse(y, oof_lgb))
print("CatBoost:", rmse(y, oof_cb))

# Train Ridge meta-learner with CV on OOFs

ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED))
ridge_cv.fit(oof_stack, y)
print("Ridge CV best alpha:", ridge_cv.alpha_)
oof_meta = ridge_cv.predict(oof_stack)
print("Stacked OOF RMSE:", rmse(y, oof_meta))

final_test_preds = ridge_cv.predict(test_stack)

# --- Save submission ---

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET_COL: final_test_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")



pd.read_csv("/kaggle/working/submission.csv")




