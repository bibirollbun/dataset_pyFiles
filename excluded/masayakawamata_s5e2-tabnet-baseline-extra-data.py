!pip install pytorch-tabnet -q


import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


%%time

train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
train_ex = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col='id')
train = pd.concat([train, train_ex], ignore_index=True)

test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')


CATS = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color']
NUMS = ['Weight Capacity (kg)']
FEATURES = CATS + NUMS
TARGET = "Price"


from sklearn.preprocessing import LabelEncoder

categorical_dims = {}
combined = pd.concat([train[FEATURES], test[FEATURES]])
new_features = FEATURES.copy()

for col in FEATURES:
    if col in NUMS:
        train[col + "_nan"] = train[col].isna().astype(int)
        test[col + "_nan"] = test[col].isna().astype(int)

        new_features.append(col + "_nan")

        median_value = train[col].median()
        train[col] = train[col].fillna(median_value)
        test[col] = test[col].fillna(median_value)
    
    if col in CATS:
        l_enc = LabelEncoder()
        combined[col] = combined[col].fillna("MISSING")
        combined[col] = l_enc.fit_transform(combined[col].astype(str).values)
        
        train[col] = combined.iloc[:len(train)][col].values
        test[col] = combined.iloc[len(train):][col].values
        
        categorical_dims[col] = len(l_enc.classes_)

FEATURES = new_features

train.head(3)


cat_idxs = [i for i, f in enumerate(FEATURES) if f in CATS]
cat_dims = [categorical_dims[f] for f in CATS]
cat_emb_dim = [min(50, (dim + 1) // 2) for dim in cat_dims]


from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


MAX_EPOCH = 50

tabnet_params = dict(
    n_d=32,
    n_a=32,
    n_steps=3,
    gamma=1.3,
    lambda_sparse=0,
    optimizer_fn=optim.Adam,
    optimizer_params=dict(lr=1e-2,weight_decay=1e-5),
    mask_type = "entmax",
    scheduler_params = dict(
        mode="min", patience=5, min_lr=1e-5, factor=0.9),
    scheduler_fn=ReduceLROnPlateau,
    seed=42,
    verbose=5,
    cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs
)

tab_reg = TabNetRegressor(**tabnet_params)


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    train[FEATURES], train[TARGET], test_size=0.2, random_state=42
)

# Pandas â†’ NumPy
X_train, X_valid = X_train.values, X_valid.values
y_train, y_valid = y_train.values.reshape(-1, 1), y_valid.values.reshape(-1, 1)


%%time

tab_reg.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['rmse'],
    max_epochs=MAX_EPOCH,
    patience=10,
    batch_size=1024*2, virtual_batch_size=128*2,
    num_workers=4,
    drop_last=False,
    loss_fn=nn.MSELoss()
)

test["PREDICTIONS"] = tab_reg.predict(test[FEATURES].values)

print(f"BEST VALID SCORE: {tab_reg.best_cost}")
print("Predictions saved in `test['PREDICTIONS']`")


sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sub['Price'] = test['PREDICTIONS'].values
sub.to_csv('submission.csv', index=False)




