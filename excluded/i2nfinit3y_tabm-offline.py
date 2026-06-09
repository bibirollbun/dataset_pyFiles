!pip install lifelines -q --no-index --find-links=/kaggle/input/cibmtr2024-import/lifelines
!pip install scikit-learn==1.4.0 -q --no-index --find-links=/kaggle/input/cibmtr2024-import/scikit_learn
!pip install rtdl_num_embeddings -q --no-index --find-links=/kaggle/input/cibmtr2024-import/rtdl_num_embeddings
!pip install delu -q --no-index --find-links=/kaggle/input/cibmtr2024-import/delu


from tabm_reference import Model, make_parameter_groups
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import rtdl_num_embeddings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, root_mean_squared_error, roc_auc_score, root_mean_squared_log_error, mean_squared_log_error
from IPython.display import clear_output
from metric import score
import warnings
warnings.filterwarnings('ignore')
import joblib
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
import delu
import math
from collections import OrderedDict
from tqdm import tqdm
from lifelines import KaplanMeierFitter

train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')



def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y

train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

train["label"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')
train.loc[train['efs']==0, 'label'] -= 0.2

train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1

combined = pd.concat([train, test], axis=0, ignore_index=True)

RMV = ["ID","efs","efs_time", "label", "y", "efs_time2"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    num_unique = combined[c].nunique()
    if num_unique < 100:
        CATS.append(c)
        train[c] = train[c].fillna(999)
        test[c] = test[c].fillna(999)
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")

NUMS = [c for c in FEATURES if not c in CATS]


# print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ", end="")
for c in FEATURES:
	
	# LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
	if c in CATS:
		print(f"{c}, ", end="")
		combined[c], _ = combined[c].factorize()
		combined[c] -= combined[c].min()
		combined[c] = combined[c].astype("int32")
		combined[c] = combined[c].astype("category")
	
	# REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
	else:
		if combined[c].dtype == "float64":
			combined[c] = combined[c].astype("float32")
		if combined[c].dtype == "int64":
			combined[c] = combined[c].astype("int32")

cat_unique = combined[CATS].nunique().to_list()

for c in NUMS:
	combined[c] = combined[c].fillna(0)

train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


cats_index = [train[FEATURES].columns.get_loc(cat) for cat in CATS]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[NUMS] = scaler.fit_transform(train[NUMS])
test[NUMS] = scaler.transform(test[NUMS])


folds = 5
train['kfold'] = -1

target = 'label'
kf = KFold(n_splits=5, random_state=42, shuffle=True)
groups = train['efs'].astype(str)
for fold, (train_idx, val_idx) in enumerate(kf.split(X=train)):
    train.loc[val_idx, 'kfold'] = fold

oof_metric = train[['kfold','ID','efs','efs_time','label','race_group']].copy()
oof_metric['prediction'] = 0.0

oof_tabm = np.zeros(train.shape[0])
test_tabm = np.zeros((5, test.shape[0]))


X_num = train[NUMS].values
X_cat = train[CATS].values

X_num_test = test[NUMS].values
X_cat_test = test[CATS].values

y = train[target].values


test_dl = DataLoader(TensorDataset(torch.tensor(X_num_test, dtype=torch.float32), torch.tensor(X_cat_test, dtype=torch.int64)), batch_size=1024, shuffle=False)

n_cont_features = len(NUMS)
n_cat_features = len(CATS)
n_classes = None
cat_cardinalities = cat_unique

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TabM
# arch_type = 'tabm'
# bins = None

# TabM-mini with the piecewise-linear embeddings.
arch_type = 'tabm-mini'


class RMSELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()
	
	def forward(self, y_pred, y_true):
		return torch.sqrt(self.mse(y_pred, y_true))


loss_fn = RMSELoss()

val_rmse_scores = []

val_cindex_scores = []
for i, (train_index, val_index) in enumerate(kf.split(train[FEATURES])):
    best = {
        "val"  : -math.inf,
        "epoch": -1,
    }
    ds_true = oof_metric.loc[oof_metric.kfold == i, ["ID", "efs", "efs_time", "race_group"]].copy().reset_index(
        drop=True)
    ds_pred = oof_metric.loc[oof_metric.kfold == i, ["ID"]].copy().reset_index(drop=True)
    
    X_num_train = X_num[train_index]
    X_cat_train = X_cat[train_index]
    y_train = y[train_index]
    
    X_num_val = X_num[val_index]
    X_cat_val = X_cat[val_index]
    y_val_all = y[val_index]
    
    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_num_train, dtype=torch.float32), torch.tensor(X_cat_train, dtype=torch.int64),
                      torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=True)
    valid_dl = DataLoader(
        TensorDataset(torch.tensor(X_num_val, dtype=torch.float32), torch.tensor(X_cat_val, dtype=torch.int64),
                      torch.tensor(y_val_all, dtype=torch.float32)), batch_size=32, shuffle=False)
    
    bins = rtdl_num_embeddings.compute_bins(torch.tensor(X_num_train, dtype=torch.float32))
    
    model = Model(
            n_num_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_classes,
            backbone={
                'type'    : 'MLP',
                'n_blocks': 3,
                'd_block' : [512, 512, 512],
                'dropout' : 0.1,
            },
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    'type'       : 'PiecewiseLinearEmbeddings',
                    'd_embedding': 64,
                    'activation' : True,
                    'version'    : 'B',
                }
            ),
            arch_type=arch_type,
            k=32,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
            # Instead of model.parameters(),
            make_parameter_groups(model),
            lr=1e-4,
            weight_decay=1e-3,
    )
    
    patience = 15
    early_stopping = delu.tools.EarlyStopping(patience, mode="max")
    
    for epoch in range(100):
        model.train()
        with tqdm(train_dl, total=len(train_dl), leave=True) as phar:
            for train_tensor in phar:
                optimizer.zero_grad()
                X_num_train, X_cat_train, y_train = [t.to(device) for t in train_tensor]
                
                output = model(X_num_train, X_cat_train).squeeze(-1)
                loss = loss_fn(output.flatten(0, 1), y_train.repeat_interleave(32))
                loss.backward()
                optimizer.step()
                
                phar.set_postfix(
                        OrderedDict(
                                epoch=f'{epoch + 1}/{100}',
                                loss=f'{loss.item():.6f}'
                        )
                )
                phar.update(1)
        
        model.eval()
        valid_pred_list = []
        for valid_tensor in valid_dl:
            X_num_val, X_cat_val, y_val = [t.to(device) for t in valid_tensor]
            with torch.no_grad():
                output = model(X_num_val, X_cat_val).squeeze(-1)
            valid_pred_list.append((output.mean(1).cpu().numpy(), y_val.cpu().numpy()))
        
        valid_pred = np.concatenate([p[0] for p in valid_pred_list])
        valid_true = np.concatenate([p[1] for p in valid_pred_list])
        val_loss = loss_fn(torch.tensor(valid_pred), torch.tensor(valid_true)).item()
        
        ds_pred["prediction"] = valid_pred
        val_cindex = score(ds_true.copy(), ds_pred.copy(), "ID")
        
        if val_cindex > best["val"]:
            print("ðŸŒ¸ New best epoch! ðŸŒ¸ with cindex: ", val_cindex)
            best = {
                "val"  : val_cindex,
                "epoch": epoch,
                'pred' : valid_pred,
            }
        
            # Inside the fold loop (after model training):
            # Save model and bins for each fold
            torch.save(model.state_dict(), f'tabm_model_{i}.pth')
        
        early_stopping.update(val_cindex)
        if early_stopping.should_stop():
            print("Early stopping")
            break
    
    oof_tabm[val_index] = best['pred']
    val_rmse = root_mean_squared_error(y_val_all, best['pred'])
    val_rmse_scores.append(val_rmse)
    
    ds_pred["prediction"] = best['pred']
    val_cindex = score(ds_true.copy(), ds_pred.copy(), "ID")
    
    val_cindex_scores.append(val_cindex)
    
    # predict test
    model.eval()
    test_pred_list = []
    with torch.no_grad():
        for test_tensor in test_dl:
            X_num_test, X_cat_test = [t.to(device) for t in test_tensor]
            output = model(X_num_test, X_cat_test).squeeze(-1)
            test_pred_list.append(output.mean(1).cpu().numpy())
    
    test_pred = np.concatenate([p for p in test_pred_list])
    test_tabm[i] = test_pred
    
    
    print(" *************************************************************************************** ")
    print("\n")
    print(f"Fold {i + 1} RMSE: {val_rmse:.6f}", f"Fold {i + 1} C-Index: {val_cindex:.6f}")
    print("\n")
    print(" *************************************************************************************** ")





print("Mean Validation RMSE: {:.6f}".format(np.mean(val_rmse_scores)))
print("Mean Validation C-Index: {:.6f}".format( np.mean(val_cindex_scores)))
print("OOF RMSE: {:.6f}".format(root_mean_squared_error(train[target], oof_tabm)))

results_df = pd.DataFrame({
        'Fold': np.arange(1, 5+1),
        'Validation RMSE': val_rmse_scores,
        'Validation C-Index': val_cindex_scores
    })


print("\n=== KFold RMSE Results ===")
print(results_df)

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_tabm
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for TabM KaplanMeier =",m)



test_mean = np.mean( test_tabm , axis=0)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = test_mean
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

"""
=== KFold RMSE Results ===
   Fold  Validation RMSE  Validation C-Index
0     1         0.239774            0.651911
1     2         0.238842            0.653592
2     3         0.237576            0.657198
3     4         0.242049            0.643281
4     5         0.565886            0.500000
5     6         0.240064            0.652549
"""

