# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/rdkit-offline/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs


import warnings
warnings.filterwarnings("ignore")
SEED = 42

train=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")


train



train.info()


targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

plt.figure(figsize= (10,8))
corr = train[targets].corr()
sns.heatmap(corr, annot=True, cmap= 'coolwarm', center= 0)
plt.title('correlations betw target variables')
plt.show()


train.isnull().sum()


def featurize(smiles, fp_bits=2048):
    s = smiles.replace("*", "C")  # polymer star -> substitute
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return np.zeros(fp_bits + 5, dtype=np.float32)
    # Morgan FP
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_bits)
    arr = np.zeros((fp_bits,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    # descriptors (small set)
    descs = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.TPSA(mol)
    ]
    return np.concatenate([arr.astype(np.float32), np.array(descs, dtype=np.float32)])

print("Featurizing train...")
X_all = np.vstack([featurize(s) for s in tqdm(train["SMILES"], desc="train feats")])
print("Featurizing test...")
X_test_all = np.vstack([featurize(s) for s in tqdm(test["SMILES"], desc="test feats")])
print("Shapes:", X_all.shape, X_test_all.shape)

# Standardize descriptors portion (last 5 cols) — helps tree models less but needed for meta-model
scaler = StandardScaler()
# scale only the descriptor tail columns to keep fingerprint distribution stable
X_all[:, -5:] = scaler.fit_transform(X_all[:, -5:])
X_test_all[:, -5:] = scaler.transform(X_test_all[:, -5:])


def select_topk_features(X, y, k=1000, seed=SEED):
    """Train a quick LGB on (X,y) and return index positions of top-k features."""
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1
    )
    model.fit(X, y)
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:k]
    return top_idx, imp


from lightgbm import early_stopping, log_evaluation
def train_ensemble_for_target(y, X, X_test, n_splits=5, top_k=1200):
    """
    Returns:
      oof_preds (len(y)), 
      stacked_test_preds (len(X_test)) where stacking is LGB/XGB/CAT averaged and meta-model fit on OOFs.
    """
    # select top features on full labeled X (to reduce noise). Use top_k or all if small.
    top_idx, imp = select_topk_features(X, y, k=min(top_k, X.shape[1]))
    X_sel = X[:, top_idx]
    X_test_sel = X_test[:, top_idx]

    # containers
    n_test = X_test_sel.shape[0]
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    test_preds_lgb = np.zeros(n_test)
    test_preds_xgb = np.zeros(n_test)
    test_preds_cat = np.zeros(n_test)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_sel, y), 1):
        X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # LightGBM (sklearn API)
        lgbm = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.02, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED + fold, n_jobs=-1
        )
        lgbm.fit(
            X_tr, y_tr,eval_set=[(X_val, y_val)], eval_metric="mae",
            callbacks=[early_stopping(100), log_evaluation(50)]
        )
        oof_lgb[val_idx] = lgbm.predict(X_val)
        test_preds_lgb += lgbm.predict(X_test_sel) / n_splits

        # XGBoost
        xgb = XGBRegressor(
            n_estimators=2000, learning_rate=0.02, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', random_state=SEED + fold, n_jobs=-1
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], eval_metric='mae',
                early_stopping_rounds=100, verbose=False)
        oof_xgb[val_idx] = xgb.predict(X_val)
        test_preds_xgb += xgb.predict(X_test_sel) / n_splits

        # CatBoost
        cat = CatBoostRegressor(
            iterations=2000, learning_rate=0.02, depth=8,
            loss_function='MAE', verbose=False, random_seed=SEED + fold
        )
        cat.fit(X_tr, y_tr, eval_set=(X_val,y_val), use_best_model=True, verbose=False)
        oof_cat[val_idx] = cat.predict(X_val)
        test_preds_cat += cat.predict(X_test_sel) / n_splits

        # print fold MAEs
        print(f"  fold {fold} MAE -> LGB {mean_absolute_error(y_val,oof_lgb[val_idx]):.5f} | "
              f"XGB {mean_absolute_error(y_val,oof_xgb[val_idx]):.5f} | "
              f"CAT {mean_absolute_error(y_val,oof_cat[val_idx]):.5f}")

    # Meta-model training: use OOF predictions as features
    # shape: (n_samples, 3)
    stack_oof = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
    stack_test = np.vstack([test_preds_lgb, test_preds_xgb, test_preds_cat]).T

    meta = Ridge(alpha=1.0, random_state=SEED)
    meta.fit(stack_oof, y)
    final_test_preds = meta.predict(stack_test)

    # For diagnostics compute blended oof MAE (on training labeled set)
    blended_oof = meta.predict(stack_oof)
    blended_oof_mae = mean_absolute_error(y, blended_oof)
    print("  >>> Blended OOF MAE:", blended_oof_mae)

    return blended_oof, final_test_preds, {
        "oof_lgb": oof_lgb, "oof_xgb": oof_xgb, "oof_cat": oof_cat,
        "test_preds_lgb": test_preds_lgb, "test_preds_xgb": test_preds_xgb, "test_preds_cat": test_preds_cat,
        "top_idx": top_idx, "importance": imp
    }


submission = pd.DataFrame({"id": test["id"].values})
oof_store = pd.DataFrame(index=train.index)

for target in targets:
    print("\n==============================")
    print("Training target:", target)
    y_full = train[target].values
    mask = ~np.isnan(y_full)
    print(f"  labeled rows: {mask.sum()} / {len(mask)}")

    if mask.sum() < 20:
        # too few samples — fallback to a simple mean predictor
        print("  Too few labels; using mean fallback.")
        submission[target] = np.repeat(np.nanmean(y_full[mask]) if mask.sum() else 0.0, X_test_all.shape[0])
        oof_store[target] = np.nan
        continue

    # train on labeled subset
    y = y_full[mask]
    X_sub = X_all[mask]
    oof_preds, test_preds, info = train_ensemble_for_target(y, X_sub, X_test_all, n_splits=5, top_k=1200)

    # store OOF in full-length vector aligned to train index
    oof_full = np.full(len(train), np.nan, dtype=float)
    oof_full[mask] = oof_preds
    oof_store[target] = oof_full

    # store test preds
    submission[target] = test_preds


submission = submission[["id"] + targets]
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")

print("\nPer-target CV MAEs (OOF):")
for target in targets:
    m = ~np.isnan(oof_store[target])
    if m.sum():
        print(f" {target}: MAE = {mean_absolute_error(train[target][m], oof_store[target][m]):.6f}")
    else:
        print(f" {target}: no OOF (fallback)")

