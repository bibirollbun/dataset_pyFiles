# =============================================================================
# HyperBoost V29.4: Definitive Final Version with All Fixes
# FIX: Removed duration-based feature leak.
# FIX: Corrected critical bugs in LOO encoding and model training.
# FIX: Corrected SimpleImputer syntax and blending typos.
# =============================================================================

import os, gc, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

@dataclass
class Config:
    DATA_PATH = "/kaggle/input/playground-series-s5e8"
    EXTERNAL_DATA_PATH = "/kaggle/input/bankdatset/bank1.csv"
    OUTPUT_NAME = "submission_legendary_v29_4.csv"
    SEED = 2025
    N_SPLITS = 10
    N_ESTIMATORS = 12000
    EARLY_STOP = 500
    USE_GPU = True
    SEEDS = (0, 42, 123) # Using multiple seeds for a robust ensemble
    MAX_FEATURES = 200
    TE_ALPHAS = [10, 20]

cfg = Config()

def seed_everything():
    os.environ["PYTHONHASHSEED"] = str(cfg.SEED)
    np.random.seed(cfg.SEED)
seed_everything()

def read_data():
    train = pd.read_csv(f"{cfg.DATA_PATH}/train.csv")
    test = pd.read_csv(f"{cfg.DATA_PATH}/test.csv")
    return train, test

def basic_clean(df):
    df = df.copy()
    for c in df.select_dtypes(include='object'):
        df[c] = df[c].astype(str).str.lower().fillna('unknown').astype('category')
    return df

# FIX: Removed duration-based features to prevent leakage.
# Integrated more robust features from previous successful pipelines.
def feature_interactions(df):
    out = df.copy()
    if 'balance' in out and 'age' in out:
        out['balance_per_age'] = out['balance'] / (out['age'].replace(0, np.nan))
        out['balance_age_interaction'] = out['balance'] * out['age']
    if 'age' in out and 'campaign' in out:
        out['age_campaign_interaction'] = out['age'] * out['campaign']
    return out

# FIX: Corrected the Leave-One-Out encoding logic to be robust and efficient.
def safe_loo_encoding(train_series, train_target):
    # Convert to numpy for speed and safety
    val = train_series.astype(str).to_numpy()
    t = train_target.to_numpy()
    
    unique_vals, inverse_indices, counts = np.unique(val, return_inverse=True, return_counts=True)
    
    sums = np.zeros(len(unique_vals))
    np.add.at(sums, inverse_indices, t)
    
    # Calculate LOO mean for each sample in the training set
    loo_means_train = (sums[inverse_indices] - t) / np.maximum(1, counts[inverse_indices] - 1)
    
    # Create a mapping from unique value to its global mean (for test/validation sets)
    global_means = sums / np.maximum(1, counts)
    mean_map = dict(zip(unique_vals, global_means))
    
    return loo_means_train, mean_map

def per_fold_encode(Xtr, Xva, Xte, ytr, cat_cols):
    te_cols = []
    for col in cat_cols:
        if col not in Xtr: continue
        
        # True LOO for the training fold
        loo_train_vals, global_mean_map = safe_loo_encoding(Xtr[col], ytr)
        Xtr[f"{col}_loo"] = loo_train_vals
        
        # Apply global means to validation and test folds
        Xva[f"{col}_loo"] = Xva[col].astype(str).map(global_mean_map).fillna(ytr.mean())
        Xte[f"{col}_loo"] = Xte[col].astype(str).map(global_mean_map).fillna(ytr.mean())
        te_cols.append(f"{col}_loo")
        
        # Bayesian
        for alpha in cfg.TE_ALPHAS:
            stats = ytr.groupby(Xtr[col].astype(str)).agg(['mean','count'])
            smooth = (stats['mean']*stats['count'] + ytr.mean()*alpha)/(stats['count']+alpha)
            for df in (Xtr,Xva,Xte):
                df[f"{col}_te{alpha}"] = df[col].astype(str).map(smooth).fillna(ytr.mean())
            te_cols.append(f"{col}_te{alpha}")
            
        for df in (Xtr,Xva,Xte):
            df.drop(columns=[col],inplace=True)
            
    return Xtr, Xva, Xte, te_cols

def select_features(X, y):
    num = X.select_dtypes(include='number')
    sel = SelectKBest(mutual_info_classif, k=min(cfg.MAX_FEATURES,num.shape[1]))
    sel.fit(num.fillna(0), y)
    keep = sel.get_feature_names_out()
    return list(keep)

def get_models():
    models = []
    for offset in cfg.SEEDS:
        seed = cfg.SEED + offset
        models.append(('lgb',lgb.LGBMClassifier(objective='binary',metric='auc',n_estimators=cfg.N_ESTIMATORS, learning_rate=0.015,num_leaves=127,subsample=0.85, colsample_bytree=0.85,random_state=seed, device='gpu' if cfg.USE_GPU else 'cpu',verbosity=-1)))
        models.append(('xgb',xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',n_estimators=cfg.N_ESTIMATORS, learning_rate=0.015,max_depth=8,subsample=0.85, colsample_bytree=0.85,random_state=seed, tree_method='gpu_hist' if cfg.USE_GPU else 'hist')))
        models.append(('cat',CatBoostClassifier(iterations=cfg.N_ESTIMATORS,learning_rate=0.015,depth=8, l2_leaf_reg=5,eval_metric='AUC',random_seed=seed, task_type='GPU' if cfg.USE_GPU else 'CPU', od_type='Iter',od_wait=cfg.EARLY_STOP,verbose=False)))
    return models

def train_and_predict(X, y, T):
    folds=StratifiedKFold(n_splits=cfg.N_SPLITS,shuffle=True,random_state=cfg.SEED)
    model_oofs=[]; model_preds=[]
    
    for name,mdl in get_models():
        print(f"\n--- Training {name} (seed: {mdl.random_state}) ---")
        oof_m=np.zeros(len(X)); pred_m=np.zeros(len(T))
        
        for fold_idx, (tr,va) in enumerate(folds.split(X,y)):
            print(f"  Fold {fold_idx+1}/{cfg.N_SPLITS}...", end="")
            Xtr,Xva=X.iloc[tr].copy(),X.iloc[va].copy()
            ytr,yva=y.iloc[tr],y.iloc[va]
            Xte=T.copy()
            
            Xtr, Xva, Xte = feature_interactions(Xtr),feature_interactions(Xva),feature_interactions(Xte)
            cat_cols=Xtr.select_dtypes(include='category').columns.tolist()
            Xtr, Xva, Xte, te_cols = per_fold_encode(Xtr,Xva,Xte,ytr,cat_cols)
            
            keep=select_features(Xtr,ytr)
            Xtr, Xva, Xte = Xtr[keep], Xva[keep], Xte[keep]
            
            # FIX: Corrected SimpleImputer syntax
            imp=SimpleImputer(strategy='median')
            Xtr = imp.fit_transform(Xtr)
            Xva = imp.transform(Xva)
            Xte = imp.transform(Xte)
            
            # FIX: Corrected model.fit calls with model-specific parameters
            if name == 'lgb':
                mdl.fit(Xtr,ytr,eval_set=[(Xva,yva)],callbacks=[lgb.early_stopping(cfg.EARLY_STOP, verbose=False)])
            elif name == 'xgb':
                mdl.fit(Xtr,ytr,eval_set=[(Xva,yva)],callbacks=[xgb.callback.EarlyStopping(rounds=cfg.EARLY_STOP, save_best=True)], verbose=False)
            elif name == 'cat':
                mdl.fit(Xtr,ytr,eval_set=[(Xva,yva)], use_best_model=True)
            
            oof_m[va]=mdl.predict_proba(Xva)[:,1]
            pred_m+=mdl.predict_proba(Xte)[:,1]/cfg.N_SPLITS
            print(" done.")
            
        model_oofs.append(oof_m); model_preds.append(pred_m)
        print(f"-> {name} (seed: {mdl.random_state}) OOF AUC: {roc_auc_score(y, oof_m):.5f}")
        
    O=np.column_stack(model_oofs); P=np.column_stack(model_preds)
    return O,P

def blend(O,y):
    w0=np.ones(O.shape[1])/O.shape[1]
    def f(w):w=np.clip(w,0,1);w/=max(1e-12, w.sum());return -roc_auc_score(y,O@w)
    # FIX: Corrected typo in bounds from O.shape[12] to O.shape[1]
    res=minimize(f,w0,method='SLSQP',bounds=[(0.01,0.8)]*O.shape[1],
                 constraints=[{'type':'eq','fun':lambda w:w.sum()-1}])
    w=res.x/res.x.sum()
    print("Optimized Blend Weights:", {i: f"{v:.4f}" for i,v in enumerate(w)})
    return w

def main():
    train,test=read_data()
    y=train['y'].astype(int)
    X=basic_clean(train.drop(['id','y'],axis=1))
    T=basic_clean(test.drop('id',axis=1))
    
    # Drop duration here, before it can be used in feature engineering
    X = X.drop(columns=['duration'], errors='ignore')
    T = T.drop(columns=['duration'], errors='ignore')

    O,P=train_and_predict(X,y,T)
    w=blend(O,y)
    
    final_oof=O@w
    print(f"\nFinal Blend OOF AUC: {roc_auc_score(y,final_oof):.5f}")
    
    iso=IsotonicRegression(out_of_bounds='clip'); iso.fit(final_oof,y)
    final_oof_cal=iso.transform(final_oof); P_cal=iso.transform(P@w)
    print(f"Calibrated OOF AUC: {roc_auc_score(y,final_oof_cal):.5f}")
    
    sub=pd.DataFrame({'id':test['id'],'y':P_cal})
    sub.to_csv(cfg.OUTPUT_NAME,index=False)
    print("Saved:",cfg.OUTPUT_NAME)
    print(sub.head())
    
if __name__=="__main__":
    main()


