!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl -q
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz -q
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl -q
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl -q
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl -q
!pip install /kaggle/input/kpytorch-tabnet/pytorch_tabnet-4.1.0-py3-none-any.whl -q


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_column', 500)
pd.set_option('display.max_row', 500)


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

train.head(3)


train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += 2*len(train)
train.y = train.y / train.y.max()
train.y = np.log( train.y )
train.y -= train.y.mean()
train.y *= -1.0


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]


combined = pd.concat([train, test], axis=0, ignore_index=True)

CATS = []
CAT_SIZE = []
CAT_EMB = []
NUMS = []

for c in FEATURES:
    if train[c].dtype == "object":
        combined[c] = combined[c].fillna("NAN")
        CATS.append(c)
    elif "age" not in c:
        combined[c] = combined[c].astype("str")
        CATS.append(c)

for c in CATS:
    combined[c], _ = combined[c].factorize(sort=True)
    combined[c] = combined[c].astype("int32")

    unique_vals = combined[c].nunique()
    CAT_SIZE.append(unique_vals + 1)  
    CAT_EMB.append(int(np.ceil(np.sqrt(unique_vals + 1))))  

for c in FEATURES:
    if c not in CATS:
        if combined[c].dtype == "float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype == "int64":
            combined[c] = combined[c].astype("int32")
        
        m = combined[c].mean()
        s = combined[c].std()
        combined[c] = (combined[c] - m) / s
        combined[c] = combined[c].fillna(0)
        
        NUMS.append(c)

train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


categorical_dims = {f: size for f, size in zip(CATS, CAT_SIZE)}
cat_dims = [categorical_dims[f] for f in CATS]
cat_emb_dim = [min(50, (dim + 1) // 2) for dim in cat_dims]
cat_idxs = [i for i, f in enumerate(FEATURES) if f in CATS]


from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR


MAX_EPOCH = 50

tabnet_params = dict(
    n_d=8,
    n_a=8,
    n_steps=2,
    gamma=1.3,
    lambda_sparse=0,
    optimizer_fn=optim.Adam,
    optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
    mask_type="entmax",
    scheduler_params=dict(
        mode="min", patience=5, min_lr=1e-5, factor=0.9),
    scheduler_fn=ReduceLROnPlateau,
    seed=42,
    verbose=5,
    cat_dims=cat_dims,
    cat_emb_dim=cat_emb_dim,
    cat_idxs=cat_idxs
)


from sklearn.model_selection import KFold
from pytorch_tabnet.tab_model import TabNetRegressor
from metric import score

REPEATS = 3
FOLDS = 5
kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)

oof_tab = np.zeros(len(train))
pred_tab = np.zeros(len(test))

for r in range(REPEATS):
    VERBOSE = r == 0
    print("#" * 30)
    print(f"### REPEAT {r+1}/{REPEATS} ###")
    print("#" * 30)
        
    for i, (train_index, val_index) in enumerate(kf.split(train)):
        print(f"\nFold {i+1}/{FOLDS}")

        X_train = train.iloc[train_index][FEATURES].values
        y_train = train.iloc[train_index]["y"].values
        X_valid = train.iloc[val_index][FEATURES].values
        y_valid = train.iloc[val_index]["y"].values
        X_test = test[FEATURES].values

        model = TabNetRegressor(**tabnet_params)
        model.fit(
            X_train, y_train.reshape(-1, 1),
            eval_set=[(X_valid, y_valid.reshape(-1, 1))],
            max_epochs=MAX_EPOCH,
            # patience=10,
            batch_size=512,
            virtual_batch_size=64,
            num_workers=0,
            drop_last=False,
        )

        oof_tab[val_index] += model.predict(X_valid).squeeze()
        pred_tab += model.predict(X_test).squeeze()

oof_tab /= REPEATS
pred_tab /= (FOLDS * REPEATS)

y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_tab
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for TabNet = {m}")


sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
sub.prediction = pred_tab
sub.to_csv('submission.csv', index=False)




