import os
import gc
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')


SEED = 42
N_FOLDS = 10
np.random.seed(SEED)


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


TARGET = "y"
ID_COL = "id"


train_orig = train.copy()
test_orig  = test.copy()


cat_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome"]
num_cols = [c for c in train.columns if c not in cat_cols + [ID_COL, TARGET]]


for c in num_cols:
    if train[c].isna().any() or test[c].isna().any():
        med = train[c].median()
        mean_val = train[c].mean()
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)


for c in cat_cols:
    if train[c].isna().any() or test[c].isna().any():
        mode_val = train[c].mode(dropna=True)[0]
        train[c] = train[c].fillna(mode_val)
        test[c]  = test[c].fillna(mode_val)


def safe_div(a, b):
    out = a / np.where(b == 0, np.nan, b)
    return np.nan_to_num(out, posinf=0.0, neginf=0.0)


month_map = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
}


if train["month"].dtype == object:
    train["month_ord"] = train["month"].str.lower().map(month_map).fillna(0).astype(int)
    test["month_ord"]  = test["month"].str.lower().map(month_map).fillna(0).astype(int)
else:
    train["month_ord"] = train["month"]
    test["month_ord"]  = test["month"]


train["pdays_unknown"] = (train["pdays"] >= 999).astype(int)
test["pdays_unknown"]  = (test["pdays"]  >= 999).astype(int)
train["pdays_capped"] = train["pdays"].clip(upper=998)
test["pdays_capped"]  = test ["pdays"].clip(upper=998)


for col in ["balance","duration","campaign","previous","pdays_capped","age","day"]:
    train[f"{col}_log1p"] = np.log1p(np.maximum(train[col], 0))
    test [f"{col}_log1p"] = np.log1p(np.maximum(test[col],  0))
    
    train[f"{col}_sqrt"] = np.sqrt(np.maximum(train[col], 0))
    test [f"{col}_sqrt"] = np.sqrt(np.maximum(test[col],  0))
    
    train[f"{col}_square"] = train[col] ** 2
    test [f"{col}_square"] = test[col] ** 2


train["dur_per_call"] = safe_div(train["duration"], np.maximum(train["campaign"], 1))
test ["dur_per_call"] = safe_div(test ["duration"], np.maximum(test ["campaign"], 1))

train["prev_contacted"] = (train["previous"] > 0).astype(int)
test ["prev_contacted"] = (test ["previous"] > 0).astype(int)

train["bal_per_age"] = safe_div(train["balance"], np.maximum(train["age"], 1))
test ["bal_per_age"] = safe_div(test ["balance"], np.maximum(test ["age"], 1))


train["duration_per_age"] = safe_div(train["duration"], np.maximum(train["age"], 1))
test ["duration_per_age"] = safe_div(test ["duration"], np.maximum(test ["age"], 1))

train["campaign_per_previous"] = safe_div(train["campaign"], np.maximum(train["previous"], 1))
test ["campaign_per_previous"] = safe_div(test ["campaign"], np.maximum(test ["previous"], 1))

train["balance_per_duration"] = safe_div(train["balance"], np.maximum(train["duration"], 1))
test ["balance_per_duration"] = safe_div(test ["balance"], np.maximum(test ["duration"], 1))


train["age_balance_interaction"] = train["age"] * train["balance"] / 10000
test ["age_balance_interaction"] = test["age"] * test["balance"] / 10000

train["duration_campaign_interaction"] = train["duration"] * train["campaign"]
test ["duration_campaign_interaction"] = test["duration"] * test["campaign"]


train["age_binned"] = pd.cut(train["age"], bins=10, labels=False)
test ["age_binned"] = pd.cut(test["age"], bins=10, labels=False)

train["balance_binned"] = pd.cut(train["balance"], bins=20, labels=False)
test ["balance_binned"] = pd.cut(test["balance"], bins=20, labels=False)

train["duration_binned"] = pd.cut(train["duration"], bins=15, labels=False)
test ["duration_binned"] = pd.cut(test["duration"], bins=15, labels=False)


numeric_features = ["age", "balance", "duration", "campaign", "previous", "pdays_capped"]
for feature in numeric_features:
    combined = pd.concat([train[feature], test[feature]])
    ranks = rankdata(combined, method='average')
    train[f"{feature}_rank"] = ranks[:len(train)]
    test[f"{feature}_rank"] = ranks[len(train):]
    
    train[f"{feature}_pct"] = train[feature].rank(pct=True)
    test[f"{feature}_pct"] = test[feature].rank(pct=True)


for c in cat_cols + ["age_binned", "balance_binned", "duration_binned"]:
    freq = pd.concat([train[c], test[c]]).value_counts(dropna=False, normalize=True)
    train[f"{c}_freq"] = train[c].map(freq)
    test [f"{c}_freq"] = test [c].map(freq)


te_cats = ["job","marital","education","contact","poutcome","month","age_binned","balance_binned","duration_binned"]


def cv_target_encode_enhanced(train_df, test_df, cat_cols, target, n_folds=5, seed=SEED, smoothing=10.0):
    train_new = train_df.copy()
    test_new  = test_df.copy()
    global_mean = train_new[target].mean()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for c in cat_cols:
        oof_te = np.zeros(len(train_new))
        test_te_folds = []
        
        for tr_idx, va_idx in skf.split(train_new, train_new[target]):
            tr, va = train_new.iloc[tr_idx], train_new.iloc[va_idx]
            stats = tr.groupby(c)[target].agg(["mean","count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            
            oof_te[va_idx] = va[c].map(smooth).fillna(global_mean).values
            test_map = test_new[c].map(smooth).fillna(global_mean).values
            test_te_folds.append(test_map)
        
        train_new[f"TE_{c}"] = oof_te
        test_new[f"TE_{c}"] = np.mean(test_te_folds, axis=0)
        
        if len(test_te_folds) > 1:
            test_te_array = np.array(test_te_folds)
            test_new[f"TE_{c}_var"] = np.var(test_te_array, axis=0)
            
            train_te_var = np.zeros(len(train_new))
            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_new, train_new[target])):
                train_te_var[va_idx] = np.var(test_te_array, axis=0)[0] if test_te_array.shape[1] > 0 else 0
            train_new[f"TE_{c}_var"] = np.mean(test_te_array.var(axis=0))
    
    return train_new, test_new


train, test = cv_target_encode_enhanced(train, test, te_cats, TARGET, n_folds=N_FOLDS, smoothing=30.0)


encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[c], test[c]], axis=0).astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))
    encoders[c] = le


drop_cols = [TARGET, ID_COL]
features = [c for c in train.columns if c not in drop_cols]


X = train[features].copy()
y = train[TARGET].astype(int).values
X_test = test[features].copy()


X = X.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

print(f"Feature count: {len(features)}")


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cb  = np.zeros(len(X))
oof_et  = np.zeros(len(X))


pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cb  = np.zeros(len(X_test))
pred_et  = np.zeros(len(X_test))


for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print("="*60)
    print(f"▶▶ Starting Fold {fold}/{N_FOLDS} ...")
    
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    print(" Training Enhanced LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=8000,  
        learning_rate=0.008, 
        num_leaves=128, 
        colsample_bytree=0.7,
        subsample=0.75,
        subsample_freq=1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=SEED,
        n_jobs=-1,
        importance_type='gain'
    )
    
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(0)]
    )
    
    oof_lgb[va_idx] = lgb_model.predict_proba(X_va)[:,1]
    pred_lgb += lgb_model.predict_proba(X_test)[:,1] / N_FOLDS
    
    print(" Training Enhanced XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=8000,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=3,
        subsample=0.75,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=-1
    )
    
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=200,
        verbose=False
    )
    
    oof_xgb[va_idx] = xgb_model.predict_proba(X_va)[:,1]
    pred_xgb += xgb_model.predict_proba(X_test)[:,1] / N_FOLDS
    
    print(" Training Enhanced CatBoost...")
    cb_model = cb.CatBoostClassifier(
        iterations=8000,
        learning_rate=0.01,
        depth=8,
        eval_metric="AUC",
        random_seed=SEED,
        l2_leaf_reg=5.0,
        verbose=False,
        loss_function="Logloss",
        early_stopping_rounds=200,
        bagging_temperature=0.8,
        border_count=128,
        feature_border_type='GreedyLogSum'
    )
    
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
    
    oof_cb[va_idx] = cb_model.predict_proba(X_va)[:,1]
    pred_cb += cb_model.predict_proba(X_test)[:,1] / N_FOLDS
    
    print(" Training Extra Trees...")
    et_model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1
    )
    
    et_model.fit(X_tr, y_tr)
    
    oof_et[va_idx] = et_model.predict_proba(X_va)[:,1]
    pred_et += et_model.predict_proba(X_test)[:,1] / N_FOLDS
    
    auc_lgb = roc_auc_score(y_va, oof_lgb[va_idx])
    auc_xgb = roc_auc_score(y_va, oof_xgb[va_idx])
    auc_cb  = roc_auc_score(y_va, oof_cb[va_idx])
    auc_et  = roc_auc_score(y_va, oof_et[va_idx])
    
    print(f" Fold {fold} completed")
    print(f"    AUCs -> LGB: {auc_lgb:.6f} | XGB: {auc_xgb:.6f} | CB: {auc_cb:.6f} | ET: {auc_et:.6f}")
    print("="*60)


auc_l = roc_auc_score(y, oof_lgb)
auc_x = roc_auc_score(y, oof_xgb)
auc_c = roc_auc_score(y, oof_cb)
auc_e = roc_auc_score(y, oof_et)

print(f"\nOOF AUCs -> LGB: {auc_l:.6f} | XGB: {auc_x:.6f} | CB: {auc_c:.6f} | ET: {auc_e:.6f}")


oof_stack_in = np.vstack([oof_lgb, oof_xgb, oof_cb, oof_et]).T
test_stack_in = np.vstack([pred_lgb, pred_xgb, pred_cb, pred_et]).T


best_w = np.array([0.25, 0.25, 0.25, 0.25])


try:
    import optuna
    
    def objective(trial):
        w1 = trial.suggest_float("w1", 0.0, 1.0)
        w2 = trial.suggest_float("w2", 0.0, 1.0)
        w3 = trial.suggest_float("w3", 0.0, 1.0)
        w4 = trial.suggest_float("w4", 0.0, 1.0)
        s = w1 + w2 + w3 + w4 + 1e-12
        w = np.array([w1/s, w2/s, w3/s, w4/s])
        blend = (oof_stack_in * w).sum(axis=1)
        return 1.0 - roc_auc_score(y, blend)
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=200, show_progress_bar=False)
    
    w = study.best_params
    s = w["w1"] + w["w2"] + w["w3"] + w["w4"] + 1e-12
    best_w = np.array([w["w1"]/s, w["w2"]/s, w["w3"]/s, w["w4"]/s])
    print(f"Optuna best weights: {best_w}")
    
except Exception as e:
    print(f"Optuna optimization failed ({e}). Using equal weights.")


blend_oof = (oof_stack_in * best_w).sum(axis=1)
blend_test = (test_stack_in * best_w).sum(axis=1)
print(f"Weighted Blend OOF AUC: {roc_auc_score(y, blend_oof):.6f}")


stacker = LogisticRegression(
    penalty="l2",
    C=0.1,
    solver="liblinear",
    max_iter=2000,
    random_state=SEED
)


stacker.fit(oof_stack_in, y)
oof_meta = stacker.predict_proba(oof_stack_in)[:,1]
test_meta = stacker.predict_proba(test_stack_in)[:,1]
print(f"Stacker (LR) OOF AUC: {roc_auc_score(y, oof_meta):.6f}")


final_oof  = 0.4 * blend_oof + 0.4 * oof_meta + 0.2 * oof_lgb  # Give more weight to best single model
final_pred = 0.4 * blend_test + 0.4 * test_meta + 0.2 * pred_lgb

print(f"\nFINAL OOF AUC: {roc_auc_score(y, final_oof):.6f}")


sub["y"] = final_pred
sub.to_csv("submission_2.csv", index=False)
print("Saved submission_2.csv")


del train, test, X, X_test
gc.collect()


print("\n" + "="*60)
print("ENHANCED MODEL COMPLETED!")
print("Key improvements made:")
print("- Increased CV folds to 10 for better stability")
print("- Added 4th model (Extra Trees) for diversity")
print("- Enhanced feature engineering (ratios, interactions, binning)")
print("- Improved target encoding with variance features")
print("- Better hyperparameter tuning")
print("- Advanced ensemble with 3-component final blend")
print("="*60)




