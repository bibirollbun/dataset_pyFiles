import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Upgrade Torch to 2.6.0+CUDA 12.4
!pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Confirm torch version
import torch
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)

import torch
print(torch.__version__)
print(torch.version.cuda)

!pip install warpgbm --no-build-isolation


from sklearn.model_selection import TimeSeriesSplit
from warpgbm import WarpGBM


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sample = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


target = 'label'
features = [ f for f in list(train) if f != target ]
all_splits = list(TimeSeriesSplit(n_splits=2, max_train_size=100_000_000, gap=1).split(train.index))


def get_best_feats(train_df):
    correlations = train_df[features].apply(lambda x: x.corr(train_df[target]))
    correlations = correlations.sort_values(ascending=False)
    best_feats = correlations.loc[ correlations.abs() > 0.025 ].index.tolist()
    return best_feats


train['pred_naive'] = 1 * np.nan
test_preds_naive = []
for split_id, (train_idx, val_idx) in enumerate( all_splits ):
    train_rows = train.index[train_idx]
    val_rows = train.index[val_idx]
    best_feats = get_best_feats( train.loc[train_rows] )
    X = train.loc[ train_rows, best_feats ].values
    y = train.loc[ train_rows, target ].values
    X_val = train.loc[ val_rows, best_feats ].values
    y_val = train.loc[ val_rows, target ].values
    model = WarpGBM(
        max_depth=10,
        num_bins=100,
        n_estimators=100,
        learning_rate=0.1,
        colsample_bytree=1.0,
        min_child_weight=4
    )
    model.fit(
        X,
        y,
        X_eval=X_val,
        y_eval=y_val,
        eval_every_n_trees=1,
        early_stopping_rounds=10,
        eval_metric="corr",
    )
    #keep best model
    best_i = int(np.argmin(model.eval_loss))
    opt_num_trees = best_i * model.eval_every_n_trees  # +1 since first eval is at tree N, not 0
    model.forest = model.forest[:( opt_num_trees + 1)]
    
    preds = model.predict(X_val)
    train.loc[ val_rows, 'pred_naive' ] = preds
    test_preds_naive.append( model.predict(test.loc[:, features ].values) )
    


train['era'] = pd.qcut(range(len(train)), q=20, labels=False, duplicates='drop')


train['pred_aware'] = 1 * np.nan
test_preds_aware = []
opt_trees = []
for split_id, (train_idx, val_idx) in enumerate( all_splits ):
    train_rows = train.index[train_idx]
    val_rows = train.index[val_idx]

    '''Define an Array of Integers to Define the Eras'''
    eras = train.loc[ train_rows, 'era' ].values
    print("Era Vector: ", eras)

    best_feats = get_best_feats( train.loc[train_rows] )
    X = train.loc[ train_rows, best_feats ].values
    y = train.loc[ train_rows, target ].values
    X_val = train.loc[ val_rows, best_feats ].values
    y_val = train.loc[ val_rows, target ].values
    model = WarpGBM(
        max_depth=10,
        num_bins=100,
        n_estimators=100,
        learning_rate=0.1,
        colsample_bytree=1.0,
        min_child_weight=4
    )
    model.fit(
        X,
        y,
        eras, # use the eras here, in .fit()
        X_eval=X_val,
        y_eval=y_val,
        eval_every_n_trees=1,
        early_stopping_rounds=10,
        eval_metric="corr",
    )
    #keep best model
    best_i = int(np.argmin(model.eval_loss))
    opt_num_trees = best_i * model.eval_every_n_trees  # +1 since first eval is at tree N, not 0
    model.forest = model.forest[:( opt_num_trees + 1 )]
    opt_trees.append(opt_num_trees)
    
    preds = model.predict(X_val)
    train.loc[ val_rows, 'pred_aware' ] = preds
    test_preds_aware.append( model.predict(test.loc[:, features ].values) )
    


print( "Naive Model Corr w/ Target:", train.loc[ :, ['pred_naive', target]].dropna().corr().iloc[0,1] )


print( "Era Aware Model Corr w/ Target:", train.loc[ :, ['pred_aware', target]].dropna().corr().iloc[0,1] )


sub_naive = sample.copy()
sub_naive['prediction'] = np.mean( test_preds_naive, axis=0)
sub_naive.to_csv('submission_naive.csv', index=False)

sub_aware = sample.copy()
sub_aware['prediction'] = test_preds_aware[1]#np.mean( test_preds_aware, axis=0)
sub_aware.to_csv('submission_aware.csv', index=False)
sub_aware.to_csv('submission.csv', index=False)


sub_naive




