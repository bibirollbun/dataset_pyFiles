import warnings
warnings.filterwarnings('ignore')


# Full end-to-end IEEE-CIS Fraud Detection pipeline (CatBoost) with heavy FE + feature selection
# Run on Kaggle: adjust PATH variable if needed.
# WARNING: heavy memory & compute. Tune down aggregations or folds if needed.
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool


PATH = "/kaggle/input/ieee-fraud-detection/"
USE_GPU = True          
N_FOLDS = 5              
SEED = 42
TOP_K_FEATURES = 200     
VERBOSE = 200

# ---------------------------
# 1) Load & merge data
# ---------------------------
train_trans = pd.read_csv(PATH + "train_transaction.csv")
train_id = pd.read_csv(PATH + "train_identity.csv")
test_trans = pd.read_csv(PATH + "test_transaction.csv")
test_id = pd.read_csv(PATH + "test_identity.csv")
sample_sub = pd.read_csv(PATH + "sample_submission.csv")

train = train_trans.merge(train_id, on="TransactionID", how="left")
test = test_trans.merge(test_id, on="TransactionID", how="left")

print("raw shapes -> train:", train.shape, "test:", test.shape)

# Save IDs and target
test_ids = test["TransactionID"].copy()
TARGET = "isFraud"
y = train[TARGET].copy()
train.drop(columns=[TARGET], inplace=True)

# Mark train/test & concat
train["isTrain"] = 1
test["isTrain"] = 0
full = pd.concat([train, test], axis=0, ignore_index=True)
del train_trans, train_id, test_trans, test_id, train, test
gc.collect()

# ---------------------------
# 2) Basic FE: time, amt, device, email, uids
# ---------------------------
# 2.1 TransactionDT derived (seconds offset -> days/hours)
full['DT_days'] = full['TransactionDT'] / (24*60*60)
full['DT_hour'] = np.floor((full['TransactionDT'] / 3600) % 24).astype(np.int16)
full['DT_weekday'] = np.floor(full['DT_days'] % 7).astype(np.int8)
full['DT_hour_sin'] = np.sin(2 * np.pi * full['DT_hour'] / 24)
full['DT_hour_cos'] = np.cos(2 * np.pi * full['DT_hour'] / 24)

# 2.2 TransactionAmt transforms
full['TransactionAmt_log'] = np.log1p(full['TransactionAmt'])
full['TransactionAmt_decimal'] = (full['TransactionAmt'] - np.floor(full['TransactionAmt'])).round(6)
full['TransactionAmt_round'] = full['TransactionAmt'].round(0)

# 2.3 Device info cleaning
def clean_device(x):
    if pd.isna(x):
        return "unknown"
    s = str(x).lower()
    if "sm-" in s or "samsung" in s: return "samsung"
    if "iphone" in s or "ios" in s: return "iphone"
    if "windows" in s: return "windows"
    if "mac os" in s or "macos" in s or "mac " in s: return "mac"
    if "linux" in s: return "linux"
    return s.split('/')[0]
full['DeviceInfo_clean'] = full['DeviceInfo'].map(clean_device)

# 2.4 Email domain groups: top domains + others
full['P_emaildomain'] = full['P_emaildomain'].fillna("unknown")
top_domains = full.loc[full['isTrain']==1, 'P_emaildomain'].value_counts().nlargest(30).index.tolist()
full['P_email_group'] = full['P_emaildomain'].where(full['P_emaildomain'].isin(top_domains), other='other')

# 2.5 UID combos
full['uid_card1_addr1'] = full['card1'].astype(str) + '_' + full['addr1'].astype(str)
full['uid_card1_addr1_email'] = full['uid_card1_addr1'] + '_' + full['P_email_group'].astype(str)
# additional combos
full['uid_card'] = full['card1'].astype(str) + '_' + full['card2'].astype(str)
full['uid_card_card3'] = full['card1'].astype(str) + '_' + full['card2'].astype(str) + '_' + full['card3'].astype(str)

# 2.6 Missing flags for important cols (identity features large set)
important_cols = ['DeviceInfo','DeviceType','P_emaildomain','R_emaildomain','card1','card2','card3','card5']
for c in important_cols:
    if c in full.columns:
        full[c + '_na'] = full[c].isna().astype(np.int8)

# 2.7 Count of missing per row
full['missing_count'] = full.isnull().sum(axis=1).astype(np.int16)

# ---------------------------
# 3) Frequency encodings (from train only)
# ---------------------------
train_mask = full['isTrain'] == 1
freq_cols = ['ProductCD','card1','card2','card3','card5','uid_card1_addr1','uid_card', 'P_email_group','DeviceInfo_clean','DeviceType']
for col in freq_cols:
    if col not in full.columns: continue
    vc = full.loc[train_mask, col].value_counts(dropna=False)
    mapping = vc.to_dict()
    full[col + '_freq'] = full[col].map(mapping).fillna(0).astype(np.int32)

# ---------------------------
# 4) Aggregations based on train only -> merge to full
# ---------------------------
# target aggregate features to compute
agg_targets = ['TransactionAmt','TransactionAmt_log','TransactionDT']
uid_keys = ['uid_card1_addr1','uid_card1_addr1_email','uid_card','uid_card_card3']
agg_funcs = ['mean','std','min','max','median','count']

for uid in uid_keys:
    if uid not in full.columns: 
        continue
    train_grp = full.loc[train_mask].groupby(uid)
    # build aggregated df
    agg_df = train_grp[agg_targets].agg(agg_funcs)
    # flatten names
    agg_df.columns = [f"{uid}_{col[0]}_{col[1]}" for col in agg_df.columns]
    agg_df = agg_df.reset_index()
    full = full.merge(agg_df, on=uid, how='left')

# fill missing aggregates for unseen uids in test
agg_cols = [c for c in full.columns if any(s in c for s in ['_mean','_std','_min','_max','_median','_count'])]
full[agg_cols] = full[agg_cols].fillna(-1)

# 4.1 Ratios: TransactionAmt / uid_mean
for uid in uid_keys:
    mean_col = f"{uid}_TransactionAmt_mean"
    if mean_col in full.columns:
        full[f"amt_div_{uid}_mean"] = full['TransactionAmt'] / (full[mean_col].replace({0:np.nan}))
        full[f"amt_div_{uid}_mean"].replace([np.inf, -np.inf], np.nan, inplace=True)
        full[f"amt_div_{uid}_mean"].fillna(-1, inplace=True)

# ---------------------------
# 5) Interaction & statistical features
# ---------------------------
# number of unique email domains per card1 in train mapped to full
if 'card1' in full.columns:
    card1_email_counts = full.loc[train_mask].groupby('card1')['P_email_group'].nunique().to_dict()
    full['card1_email_unique'] = full['card1'].map(card1_email_counts).fillna(0).astype(np.int16)

# per-card transaction counts (train)
if 'card1' in full.columns:
    card1_txn_counts = full.loc[train_mask].groupby('card1').size().to_dict()
    full['card1_txn_count'] = full['card1'].map(card1_txn_counts).fillna(0).astype(np.int32)

# ---------------------------
# 6) Cleanup & drop columns not needed
# ---------------------------
drop_cols = ['TransactionID','TransactionDT','DeviceInfo']  # drop raw DT, raw DeviceInfo, keep cleaned
for c in drop_cols:
    if c in full.columns:
        full.drop(columns=[c], inplace=True)

# get train/test back
train_fe = full[full['isTrain']==1].copy().reset_index(drop=True)
test_fe = full[full['isTrain']==0].copy().reset_index(drop=True)
train_fe.drop(columns=['isTrain'], inplace=True)
test_fe.drop(columns=['isTrain'], inplace=True)

# put back target to train_fe
train_fe[TARGET] = y.values
print("After FE shapes -> train:", train_fe.shape, " test:", test_fe.shape)
gc.collect()

# ---------------------------
# 7) Identify categorical columns for CatBoost
# ---------------------------
# CatBoost can take list of categorical column names. We'll choose object-type plus small-cardinality columns we created.
possible_cat = []
for col in train_fe.columns:
    if col == TARGET: 
        continue
    # keep object dtypes
    if train_fe[col].dtype == 'object':
        possible_cat.append(col)
    # also include known categorical-like cols
    if col in ['ProductCD','card4','card6','P_email_group','R_emaildomain','DeviceType','DeviceInfo_clean',
               'uid_card1_addr1','uid_card1_addr1_email','uid_card','uid_card_card3']:
        if col in train_fe.columns:
            possible_cat.append(col)
# ensure uniqueness
cat_features = [c for c in pd.unique(possible_cat) if c in train_fe.columns]
print("Cat features chosen:", len(cat_features), "sample:", cat_features[:20])

# Convert object dtype to string (CatBoost can accept strings)
for c in cat_features:
    train_fe[c] = train_fe[c].astype(str)
    test_fe[c] = test_fe[c].astype(str)

# ---------------------------
# 8) Quick feature list & sanity
# ---------------------------
features_all = [c for c in train_fe.columns if c != TARGET]
print("Total candidate features:", len(features_all))

# ---------------------------
# 9) Quick first-pass CatBoost training to get importances
#    Use a light CV (3-fold stratified) to get stable importances
# ---------------------------
def cb_train_get_importance(X, y, cat_features, n_splits=3, params=None, seed=SEED):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    feat_imp = pd.DataFrame(index=features_all)
    oof = np.zeros(len(X))
    for i, (tr_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"Importance fold {i+1}/{n_splits}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        cb = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.05,
            depth=6,
            eval_metric='AUC',
            random_seed=seed+i,
            verbose=False,
            task_type='GPU' if USE_GPU else 'CPU'
        )
        cb.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, use_best_model=True)
        oof[val_idx] = cb.predict_proba(X_val)[:,1]
        imp = pd.Series(cb.get_feature_importance(train_pool), index=features_all)
        feat_imp[f'fold_{i+1}'] = imp
        gc.collect()
    feat_imp['avg'] = feat_imp.mean(axis=1)
    print("OOF AUC (first pass):", roc_auc_score(y, oof))
    return feat_imp.sort_values('avg', ascending=False), oof

# Run first pass importance (may take time)
feat_imp, oof_first = cb_train_get_importance(train_fe[features_all], train_fe[TARGET].values, cat_features, n_splits=3)
print("Top 30 features by importance:\n", feat_imp['avg'].head(30))

# ---------------------------
# 10) Feature selection: keep top K features (or all if TOP_K_FEATURES is None)
# ---------------------------
if TOP_K_FEATURES is not None and TOP_K_FEATURES < len(feat_imp):
    top_feats = feat_imp['avg'].nlargest(TOP_K_FEATURES).index.tolist()
else:
    top_feats = features_all.copy()

# Ensure categorical features selected are in top_feats
cat_features_final = [c for c in cat_features if c in top_feats]
print(f"Selected top features: {len(top_feats)}; categorical among them: {len(cat_features_final)}")

# ---------------------------
# 11) Final training with K-fold CV and OOF + test predictions
# ---------------------------
oof = np.zeros(len(train_fe))
test_preds = np.zeros(len(test_fe))
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(folds.split(train_fe[top_feats], train_fe[TARGET])):
    print(f"\nTraining fold {fold+1}/{N_FOLDS}")
    X_tr, X_val = train_fe.loc[tr_idx, top_feats], train_fe.loc[val_idx, top_feats]
    y_tr, y_val = train_fe.loc[tr_idx, TARGET].values, train_fe.loc[val_idx, TARGET].values

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features_final)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_final)

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        eval_metric='AUC',
        random_seed=SEED+fold,
        early_stopping_rounds=200,
        verbose=VERBOSE,
        task_type='GPU' if USE_GPU else 'CPU'
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    val_pred = model.predict_proba(X_val)[:,1]
    oof[val_idx] = val_pred
    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

    # predict test
    test_preds += model.predict_proba(test_fe[top_feats])[:,1] / N_FOLDS

    # free memory per fold
    del model, train_pool, val_pool, X_tr, X_val
    gc.collect()

print("\nOverall OOF AUC:", roc_auc_score(train_fe[TARGET].values, oof))

# ---------------------------
# 12) Create submission
# ---------------------------
submission = pd.DataFrame({
    "TransactionID": test_ids,
    "isFraud": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

# ---------------------------
# 13) Save top features & importance (optional)
# ---------------------------
feat_imp_df = feat_imp.reset_index().rename(columns={'index':'feature'}).sort_values('avg', ascending=False)
feat_imp_df.to_csv("feature_importances_firstpass.csv", index=False)
print("Saved feature_importances_firstpass.csv")

# Done


