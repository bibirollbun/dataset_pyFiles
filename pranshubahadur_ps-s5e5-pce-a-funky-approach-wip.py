# ================================================================
#  Cat-First Decision Pipeline  (Lean FE · Sex-split LGB · CatBoost)
# ================================================================
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_log_error

# ---------------- CONFIG ----------------------------------------
N_SPLITS       = 5
N_AGE_BINS     = 16
MIN_GROUP_SIZE = 100
EPS            = 0.002          # tolerance Cat vs LGB

LGB_PAR = dict(
    objective='rmse', metric='rmse', learning_rate=0.05,
    n_estimators=2000, max_depth=11, device='gpu'
)

CAT_PAR = dict(
    iterations=2000, learning_rate=0.05, depth=8,
    l2_leaf_reg=3, task_type='GPU', loss_function='RMSE',
    eval_metric='RMSE', random_seed=42, verbose=100
)

# ---------------- LOAD & FEATURES -------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test_ids = test.id.values

def preprocess(df):
    #df["AgeGroup"]      = pd.cut(df["Age"], bins=[0,25,40,60,80,100], labels=False)
    #df["BodyGroup"]     = pd.qcut(df["Height"]*df["Weight"], q=5, labels=False)
    #df["Sex_AgeGroup"]  = df["Sex"] + "_" + df["AgeGroup"].astype(str)
    #df["Sex_BodyGroup"] = df["Sex"] + "_" + df["BodyGroup"].astype(str)
    df["Height_Weight"] = df["Height"]*df["Weight"]
    df["HeartRate_Temp"]= df["Heart_Rate"]*df["Body_Temp"]
    df["Weight_Temp"]   = df["Weight"]   *df["Body_Temp"]
    df["Duration_Age"]  = df["Duration"] *df["Age"]
    df["HR_per_min"]    = df["Heart_Rate"] / df["Duration"]
    df["BMI"]           = df["Weight"] / ((df["Height"]/100)**2)
    return df

train = preprocess(train)
test = preprocess(test)

kb = KBinsDiscretizer(N_AGE_BINS, encode='ordinal', strategy='quantile')
train["AgeQ"]=kb.fit_transform(train[["Age"]]).astype(int)
test["AgeQ"] =kb.transform   (test[["Age"]]).astype(int)
for df in (train,test):
    df["Sex"]=df.Sex.astype("category"); df["AgeQ"]=df.AgeQ.astype("category")

CAT_COLS=["Sex","AgeQ"]; DROP=["id","Calories"]
y_true=train.Calories.values; logy=np.log1p(y_true)

# ---------------- OOF HOLDERS -----------------------------------
N=len(train)
oof_final=np.zeros(N)        # will hold row-wise chosen prediction
decision={}                  # (fold,sex,age_bin) -> 'cat' or 'lgb'

# stratified folds
strat=train.Sex.astype(str)+"_"+train.AgeQ.astype(str)
folds=list(StratifiedKFold(N_SPLITS,shuffle=True,random_state=42).split(train,strat))

# ---------------- CV LOOP ---------------------------------------
for fold,(tr_idx,val_idx) in enumerate(folds,1):
    print(f"\n── Fold {fold}")
    tr,val=train.iloc[tr_idx],train.iloc[val_idx]
    X_tr,X_val=tr.drop(columns=DROP),val.drop(columns=DROP)
    y_tr,y_val=np.log1p(tr.Calories),np.log1p(val.Calories)

    # ❶ CatBoost GLOBAL  (train FIRST)
    cb=CatBoostRegressor(**CAT_PAR)
    cb.fit(X_tr,y_tr,eval_set=(X_val,y_val),cat_features=CAT_COLS,use_best_model=True)
    cat_pred=np.expm1(cb.predict(X_val))

    # ❷ LightGBM SEX-globals + optional AgeQ heads
    lgb_pred=np.zeros(len(val))
    for sex in ["male","female"]:
        mtr=tr.Sex==sex; mva=val.Sex==sex
        if not mva.any(): continue

        lg=lgb.LGBMRegressor(**LGB_PAR, early_stopping_rounds=100)
        lg.fit(X_tr[mtr],y_tr[mtr],
               eval_set=[(X_val[mva],y_val[mva])],
               categorical_feature=CAT_COLS,
               )
        lgb_pred[mva]=np.expm1(lg.predict(X_val[mva]))

        # AgeQ sub-heads
        for age_bin in tr.AgeQ.cat.categories:
            s_tr=mtr&(tr.AgeQ==age_bin); s_va=mva&(val.AgeQ==age_bin)
            if s_tr.sum()<MIN_GROUP_SIZE or not s_va.any(): continue
            sub=lgb.LGBMRegressor(**LGB_PAR, early_stopping_rounds=100)
            sub.fit(X_tr[s_tr],y_tr[s_tr],
                    eval_set=[(X_val[s_va],y_val[s_va])],
                    categorical_feature=CAT_COLS,
                    )
            sub_pred=np.expm1(sub.predict(X_val[s_va]))

            # Compare RMSEs
            rm_sub=mean_squared_log_error(val.Calories[s_va],sub_pred,squared=False)
            rm_glb=mean_squared_log_error(val.Calories[s_va],lgb_pred[s_va],squared=False)
            # choose best LGB for this slice
            best_lgb_pred=sub_pred if rm_sub+EPS<rm_glb else lgb_pred[s_va]

            # Compare Cat vs chosen LGB
            rm_cat=mean_squared_log_error(val.Calories[s_va],cat_pred[s_va],squared=False)
            use_cat = rm_cat+EPS < rm_sub and rm_cat+EPS < rm_glb
            print(sex, age_bin, rm_sub, rm_glb, rm_cat)

            if use_cat:
                abs_idx = val.index[s_va]          # absolute row positions in full train
                oof_final[abs_idx] = cat_pred[s_va]

                decision[(fold,sex,int(age_bin))]='cat'
            else:
                abs_idx = val.index[s_va] 
                oof_final[abs_idx]=best_lgb_pred
                decision[(fold,sex,int(age_bin))]='lgb'

    # rows not covered by any AgeQ head fall back to Cat vs LGB global
    uncovered=np.where(oof_final[val_idx]==0)[0]
    for idx in uncovered:
        i=val_idx[idx]
        sex=val.iloc[idx].Sex; age_bin=int(val.iloc[idx].AgeQ)
        rm_cat = mean_squared_log_error([y_true[i]],[cat_pred[idx]],squared=False)
        rm_lgb = mean_squared_log_error([y_true[i]],[lgb_pred[idx]],squared=False)
        if rm_cat+EPS<rm_lgb:
            oof_final[i]=cat_pred[idx]; decision[(fold,sex,age_bin)]='cat'
        else:
            oof_final[i]=lgb_pred[idx]; decision[(fold,sex,age_bin)]='lgb'

print("\nOOF RMSLE:",mean_squared_log_error(y_true,oof_final,squared=False))

# --------- Consolidate slice decision across folds -------------
from collections import defaultdict
vote=defaultdict(list)
for (fold,sex,bin_),flag in decision.items():
    vote[(sex,bin_)].append(flag)
final_decision={k:max(set(v),key=v.count) for k,v in vote.items()}  # majority vote

# --------- FULL-DATA REFIT --------------------------------------
X_all=train.drop(columns=DROP); y_all=np.log1p(train.Calories)
# CatBoost global full
cb_full=CatBoostRegressor(**CAT_PAR)
Xa,Xv,ya,yv=train_test_split(X_all,y_all,test_size=0.10,random_state=42,stratify=strat)
cb_full.fit(Xa,ya,eval_set=(Xv,yv),cat_features=CAT_COLS,use_best_model=True)
cat_test=np.expm1(cb_full.predict(test.drop(columns=["id"])))

# LightGBM sex-globals
lgb_test=np.zeros(len(test))
for sex in ["male","female"]:
    mask_t=train.Sex==sex; mask_te=test.Sex==sex
    Xa,Xv,ya,yv=train_test_split(
        X_all[mask_t],y_all[mask_t],test_size=0.10,
        random_state=42,stratify=train.AgeQ[mask_t])
    lg=lgb.LGBMRegressor(**LGB_PAR, early_stopping_rounds=100)
    lg.fit(Xa,ya,eval_set=[(Xv,yv)],categorical_feature=CAT_COLS,
           )
    lgb_test[mask_te]=np.expm1(lg.predict(test.loc[mask_te].drop(columns=["id"])))

    # sub-heads
    for age_bin in train.AgeQ.cat.categories:
        sb_tr=mask_t&(train.AgeQ==age_bin); sb_te=mask_te&(test.AgeQ==age_bin)
        if sb_tr.sum()<MIN_GROUP_SIZE or not sb_te.any(): continue
        Xa,Xv,ya,yv=train_test_split(
            X_all[sb_tr],y_all[sb_tr],test_size=0.10,random_state=42)
        sub=lgb.LGBMRegressor(**LGB_PAR, early_stopping_rounds=50)
        sub.fit(Xa,ya,eval_set=[(Xv,yv)],categorical_feature=CAT_COLS,
                )
        lgb_sub_pred=np.expm1(sub.predict(test.loc[sb_te].drop(columns=["id"])))

        flag=final_decision.get((sex,int(age_bin)),'lgb')
        if flag=='cat':
            lgb_test[sb_te]=cat_test[sb_te]     # overwrite with Cat
        else:
            # choose best LGB variant
            lgb_test[sb_te]=lgb_sub_pred

# Rows without explicit decision: choose Cat if Cat beat LGB global slice in CV
for idx,row in test.iterrows():
    if lgb_test[idx]==0:
        sex=row.Sex; age_bin=int(row.AgeQ)
        flag=final_decision.get((sex,age_bin),'lgb')
        lgb_test[idx]=cat_test[idx]

# Final clip & save
lo,hi=np.quantile(y_true,[0.01,0.99])
final=np.clip(lgb_test,lo,hi)
pd.DataFrame({"id":test_ids,"Calories":final}).to_csv("submission.csv",index=False)
print("\n✅ submission.csv saved.")





