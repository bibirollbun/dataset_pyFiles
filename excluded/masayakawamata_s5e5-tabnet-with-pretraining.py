# ============================================================
# TabNet (Pretraining → Supervised fine-tune) + K-fold RMSLE
# ============================================================
!pip install -q pytorch-tabnet 

import gc, itertools, pickle, warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model     import TabNetRegressor
import torch

SEED, FOLDS = 42, 5
TARGET, ID_COL = "Calories", "id"
THR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def rmsle(y, y_hat):
    return np.sqrt(mean_squared_log_error(
        np.maximum(0, y), np.maximum(0, y_hat)
    ))

# ------------------------------------------------------------
# 1. Data & light feature engineering
# ------------------------------------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

train['Sex'] = train['Sex'].map({'male':0,'female':1})
test['Sex'] = test['Sex'].map({'male':0,'female':1})

X_full = train.drop([ID_COL, TARGET], axis=1)
y_full = np.log1p(train[TARGET].values).reshape(-1,1)  # TabNet expects 2-D
X_test = test.drop([ID_COL], axis=1)

cat_cols  = ["Sex"]
cat_idxs  = [X_full.columns.get_loc(c) for c in cat_cols]
cat_dims  = [X_full[c].nunique() for c in cat_cols]

# ------------------------------------------------------------
# 2. Unsupervised pre-training on ALL rows (train+test)
# ------------------------------------------------------------
unsup_data = pd.concat([X_full, X_test], axis=0).values
pretrainer = TabNetPretrainer(
    input_dim = X_full.shape[1],
    cat_idxs  = cat_idxs,
    cat_dims  = cat_dims,
    cat_emb_dim=2,
    n_d = 16, n_a = 16, n_steps=3,
    optimizer_fn = torch.optim.Adam,
    optimizer_params = dict(lr=1e-3),
    mask_type="entmax",
    seed=SEED,
    device_name=THR_DEVICE,
)

print(">> Pretraining TabNet (unsupervised)…")
pretrainer.fit(
    X_train = unsup_data,
    eval_set = [unsup_data],
    max_epochs = 50,
    patience = 5,
    batch_size = 8192,
    virtual_batch_size = 256,
    num_workers = 0,
    drop_last = False,
)

# ------------------------------------------------------------
# 3. Supervised fine-tuning with K-fold CV
# ------------------------------------------------------------
kf = KFold(FOLDS, shuffle=True, random_state=SEED)
oof_log  = np.zeros(len(train))
test_log = np.zeros(len(test))

for fold,(tr,vl) in enumerate(kf.split(X_full),1):
    reg = TabNetRegressor(
        input_dim  = X_full.shape[1],
        output_dim = 1,
        cat_idxs   = cat_idxs,
        cat_dims   = cat_dims,
        cat_emb_dim=2,
        n_d=16, n_a=16, n_steps=3,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        mask_type="entmax",
        seed=SEED,
        device_name=THR_DEVICE,
    )

    reg.fit(
        X_train = X_full.iloc[tr].values, y_train = y_full[tr],
        eval_set=[(X_full.iloc[vl].values, y_full[vl])],
        max_epochs=400,
        patience=30,
        batch_size=4096,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False,
        from_unsupervised=pretrainer
    )

    oof_log[vl] = reg.predict(X_full.iloc[vl].values).ravel()
    test_log   += reg.predict(X_test.values).ravel() / FOLDS
    fold_r = rmsle(np.expm1(y_full[vl].ravel()), np.expm1(oof_log[vl]))
    print(f"Fold {fold}: RMSLE={fold_r:.5f}")

    del reg; gc.collect()

oof_pred  = np.expm1(oof_log).clip(0)
test_pred = np.expm1(test_log).clip(0)
print(f"\n=== Full OOF RMSLE: {rmsle(train[TARGET], oof_pred):.5f} ===")

# ------------------------------------------------------------
# 4. Save artefacts
# ------------------------------------------------------------
pickle.dump(oof_pred , open("oof_pred.pkl","wb"))
pickle.dump(test_pred, open("test_pred.pkl","wb"))
pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}) \
  .to_csv("submission.csv", index=False)

print("\nSaved: oof_pred.pkl, test_pred.pkl, submission.csv")




