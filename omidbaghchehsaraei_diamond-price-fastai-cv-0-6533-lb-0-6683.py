import gc
import torch
import random
import warnings
import numpy as np
import pandas as pd
from fastai.tabular.all import *
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from fastai.callback.all import SaveModelCallback, EarlyStoppingCallback

warnings.filterwarnings('ignore') 

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)


train_df = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/train.csv")
test_df = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/test.csv") 

TARGET = 'price'

features = [col for col in test_df.columns if col != 'id'] 
all_cat_names = test_df.select_dtypes(include=['object']).columns.tolist() 
all_cont_names = [col for col in features if col not in all_cat_names] 


reg_procs = [Categorify, FillMissing, Normalize]

layers = [800, 400, 200]
wd_values = [0.001, 0.01, 0.1] 
best_r2 = -float('inf')
best_wd = None
best_lr = 0.01

for wd_val in wd_values:
    
    print(f"\n--- Tuning with Weight Decay (wd): {wd_val} ---")

    to = TabularPandas(
        train_df,
        procs=reg_procs,
        cat_names=all_cat_names,
        cont_names=all_cont_names,
        y_names=TARGET,
        y_block=RegressionBlock(),
        splits=RandomSplitter(valid_pct=0.2, seed=42)(train_df)
    )
    dls = to.dataloaders(bs=512)

    learn = tabular_learner(dls, layers=layers, metrics=R2Score(), opt_func=Adam, wd=wd_val)
    lr_finder = learn.lr_find()
    # Find the learning rate where the loss is the steepest
    chosen_lr = lr_finder.valley
    
    learn = tabular_learner(
        dls,
        layers=layers,
        metrics=R2Score(),
        opt_func=Adam,
        wd=wd_val,
        cbs=[
            SaveModelCallback(monitor='r2_score', comp=np.greater, fname='temp_best_model'),
            EarlyStoppingCallback(monitor='valid_loss', patience=15),
        ]
    )
    
    learn.fit_one_cycle(10, lr_max=chosen_lr)

    val_preds, val_targs = learn.get_preds(dl=dls.valid)
    tuning_r2 = r2_score(to_np(val_targs).squeeze(), to_np(val_preds).squeeze())
    
    print(f"Validation R2 Score for wd={wd_val}: {tuning_r2:.6f}")
    
    if tuning_r2 > best_r2:
        best_r2 = tuning_r2
        best_wd = wd_val
        best_lr = chosen_lr

    del learn, dls, to
    gc.collect()

print("\n==================================================") 
print(f"Best Weight Decay (wd): {best_wd}")
print(f"Best Learning Rate (lr): {best_lr}")
print(f"Best Validation R2 Score: {best_r2:.6f}")
print("==================================================")


oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))
r2_scores_per_fold = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
    
    print(f"\n--- Training Fold {fold+1}/5 ---")
    
    to = TabularPandas(
        train_df,
        procs=reg_procs,
        cat_names=all_cat_names,
        cont_names=all_cont_names,
        y_names=TARGET,
        y_block=RegressionBlock(),
        splits=(list(train_idx), list(valid_idx))
    )

    dls = to.dataloaders(bs=512)

    learn = tabular_learner(
        dls,
        layers=layers,
        metrics=R2Score(),
        opt_func=Adam,
        wd=best_wd,
        cbs=[
            SaveModelCallback(monitor='r2_score', comp=np.greater, fname='best_model'),
            EarlyStoppingCallback(monitor='valid_loss', patience=15),
        ]
    )
    
    learn.fit_one_cycle(40, lr_max=best_lr)
    
    learn.load('best_model')
    
    val_preds, val_targs = learn.get_preds(dl=dls.valid)
    fold_r2 = r2_score(to_np(val_targs).squeeze(), to_np(val_preds).squeeze())
    r2_scores_per_fold.append(fold_r2)
    
    oof_preds[valid_idx] = to_np(val_preds).squeeze()

    test_dl = dls.test_dl(test_df)
    fold_test_preds, _ = learn.get_preds(dl=test_dl)
    test_preds += to_np(fold_test_preds).squeeze() / kf.n_splits
    
    del learn, dls, to
    gc.collect()


mean_r2 = np.mean(r2_scores_per_fold)
print("\n==================================================")
print(f"Mean R2 Score per Fold (with tuned params): {mean_r2:.5f}")
print("==================================================")

overall_oof_r2 = r2_score(train_df[TARGET], oof_preds) 
print("\n==================================================")
print(f"Overall Out-of-Fold R2 Score (with tuned params): {overall_oof_r2:.5f}")
print("==================================================") 


oof_df = pd.DataFrame({
    'id': train_df['id'],
    'price': oof_preds
}) 
oof_df.to_csv('oof.csv', index=False) 

submission = pd.read_csv("/kaggle/input/predicting-the-price-of-diamond/submission.csv")
submission[TARGET] = test_preds
submission.to_csv('submission.csv', index=False)
submission.head()

