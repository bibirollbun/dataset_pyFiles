import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import normaltest, skew
import json


df_train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
df_train_transaction.head()


df_train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
df_train_identity.head()


#left merge to preserve all rows in df_train_transaction 
df_train = pd.merge(
    df_train_transaction,
    df_train_identity,
    on="TransactionID",
    how="left"
)
df_train.head()


df_train.shape


def _missing_pattern_key(args): # map expects function to have 1 argument only
    col, col_mask = args
    return (col, tuple(col_mask))

def find_always_missing_together_parallel(df):
    np.random.seed(42)
    random_idx = np.random.choice(df.index, size=50000, replace=False) 
    # taking 50k to avoid too much time
    sorted_idx = np.sort(random_idx)
    df_sample = df.loc[sorted_idx]
    mask = df_sample.isna()
    mask = mask.loc[:, (mask.sum() > 0) & (mask.sum() < len(mask))]
    #removes columns that are always missing or always have values in it
    cols = mask.columns
    with Pool(cpu_count()) as pool:
        col_patterns = pool.map(_missing_pattern_key, [(col, mask[col].values) for col in cols])
        # pool assigns each worker with one column
    pattern_groups = defaultdict(list)
    for col, pattern in col_patterns:
        pattern_groups[pattern].append(col)
    always_together = [group for group in pattern_groups.values() if len(group) > 1]
    #keeps only those lists that contain more than one column (i.e., columns that always go missing together).
    return always_together

always_missing_together = find_always_missing_together_parallel(df_train)
for group in always_missing_together:
    print("Columns always missing together:", group)


categorical_cols = df_train.select_dtypes(
    include=['object','category']).columns.tolist()
print(categorical_cols)


def alphabetic_label_encode(df, categorical_cols):
    df_encoded = df.copy()
    mappings = {}
    for col in categorical_cols:
        categories = sorted(df_encoded[col].dropna().unique())
        mapping = {cat: i+1 for i, cat in enumerate(categories)}
        mappings[col] = mapping
        df_encoded[col] = df_encoded[col].map(mapping)
    
    # Save mappings to JSON
    with open('label_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    
    return df_encoded

df_train = alphabetic_label_encode(df_train, categorical_cols)


def agglomerate_missing_groups(df, groups, categorical_cols, n_clusters=1):
    df_new = df.copy()
    agg_col_names = []
    for i, group in enumerate(groups, 1):
        # Filter out categorical columns from the group
        numeric_group = [col for col in group if col not in categorical_cols]
        
        # Skip if no numeric columns in group
        if not numeric_group:
            continue
            
        mask = ~df[numeric_group].isna().all(axis=1)
        X = df.loc[mask, numeric_group].fillna(df[numeric_group].mean())
        agg = FeatureAgglomeration(n_clusters=n_clusters)
        reduced = agg.fit_transform(X)
        col_name = f"A{i}"
        agg_col_names.append(col_name)
        df_new[col_name] = np.nan
        df_new.loc[mask, col_name] = reduced.ravel()
        df_new = df_new.drop(columns=numeric_group)
    return df_new

df_train_reduced = agglomerate_missing_groups(df_train, always_missing_together, categorical_cols)
df_train_reduced.head()


def add_nan_indicators_and_convert(df):
    df_new = df.copy()
    nan_indicators = {}
    
    for col in df_new.columns:
        if col in ['TransactionID', 'isFraud'] or col.endswith('_nan'):
            continue
        nan_indicators[f"{col}_nan"] = df_new[col].isna().astype(int)
    
    indicators_df = pd.DataFrame(nan_indicators, index=df_new.index)
    cols = []
    for col in df_new.columns:
        cols.append(col)
        if f"{col}_nan" in indicators_df.columns:
            cols.append(f"{col}_nan")
    
    combined = pd.concat([df_new, indicators_df], axis=1)
    combined = combined[cols]
    
    # Convert all columns to numeric, coerce errors, fill NaN with 0
    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
    combined = combined.fillna(0)
    return combined

df_train_reduced = add_nan_indicators_and_convert(df_train_reduced)
df_train_reduced.head()


def normaltest_and_skew_report(df):
    results = []
    skip_cols = {'TransactionID', 'isFraud'}
    numeric_cols = [col for col in df.select_dtypes(include='number').columns if col not in skip_cols and not col.endswith('_nan')]
    for col in numeric_cols:
        nan_indicator_col = f"{col}_nan"
        # Only use rows where indicator is 0 OR indicator col does not exist
        if nan_indicator_col in df.columns:
            data = df.loc[df[nan_indicator_col] == 0, col].dropna()
        else:
            data = df[col].dropna()
        if len(data) < 8:
            continue
        stat, p = normaltest(data)
        col_skew = skew(data)
        results.append({
            'column': col,
            'normaltest_stat': stat,
            'normaltest_pvalue': p,
            'skewness': col_skew
        })
    return pd.DataFrame(results)

report = normaltest_and_skew_report(df_train_reduced)
print(report)


def get_normal_columns(report_df, alpha=0.05):
    """
    Returns a list of columns that pass the normality test (p >= alpha).
    """
    return report_df.loc[report_df['normaltest_pvalue'] >= alpha, 'column'].tolist()

normal_columns = get_normal_columns(report)
print(normal_columns)


df_test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
df_test_transaction.head()


df_test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
df_test_identity.head()


df_test = pd.merge(
    df_test_transaction,
    df_test_identity,
    on="TransactionID",
    how="left"
)
df_test.head()


with open('label_mappings.json', 'r') as f:
    mappings = json.load(f)

for col in categorical_cols:
    if col in mappings:
        df_test[col] = df_test[col].map(mappings[col])

df_test.head()


df_test_reduced = agglomerate_missing_groups(df_test, always_missing_together, categorical_cols)
df_test_reduced.head()


df_test_reduced = add_nan_indicators_and_convert(df_test_reduced)
df_test_reduced.head()


df_test_reduced.head()


df_train_reduced.to_csv('df_train_reduced.csv', index=False)
df_test_reduced.to_csv('df_test_reduced.csv', index=False)


# full_pipeline_with_graph.py
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

# ------------------ Load Data ------------------
TRAIN_PATH = 'df_train_reduced.csv'
TEST_PATH  = 'df_test_reduced.csv'
OUT_PATH   = 'lgbm_submission.csv'

TARGET = 'isFraud'
IDCOL = 'TransactionID'
RANDOM_STATE = 42
N_SPLITS = 5

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

features = [c for c in train.columns if c not in [TARGET, IDCOL]]

X = train[features]
y = train[TARGET].astype(int)
X_test = test[features]

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ------------------ Logistic Regression ------------------
print("---- Logistic Regression baseline ----")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=5000, class_weight='balanced', n_jobs=-1)
auc_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print("CV ROC AUC:", auc_scores, "Mean:", np.mean(auc_scores))


# ------------------ Random Forest ------------------
print("---- RandomForest quick ----")
rf = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    class_weight='balanced',
    random_state=RANDOM_STATE
)

rf_auc = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print("RF CV ROC AUC:", rf_auc, "Mean:", np.mean(rf_auc))


# ------------------ RF ROC-AUC Graph ------------------
plt.figure(figsize=(7,5))
plt.plot(range(1, len(rf_auc)+1), rf_auc, marker='o')
plt.title("Random Forest CV ROC AUC (per fold)")
plt.xlabel("Fold")
plt.ylabel("ROC AUC")
plt.grid(True)
plt.show()


# ------------------ LightGBM OOF Training ------------------
print("---- LightGBM OOF training ----")
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importance_df = pd.DataFrame()

# LightGBM params
lgb_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "n_estimators": 10000,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
    "n_jobs": -1
}

# class imbalance fix
neg = (y == 0).sum()
pos = (y == 1).sum()
lgb_params['scale_pos_weight'] = neg / pos

folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y), 1):
    print(f"\n>>> Fold {fold}")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set)

    clf = lgb.train(
        lgb_params,
        train_set,
        valid_sets=[train_set, val_set],
        valid_names=['train', 'valid'],
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(200)
        ]
    )

    best_iter = clf.best_iteration
    oof_preds[val_idx] = clf.predict(X_val, num_iteration=best_iter)
    test_preds += clf.predict(X_test, num_iteration=best_iter) / N_SPLITS

    # feature importance
    fold_imp = pd.DataFrame({
        "feature": features,
        "importance": clf.feature_importance("gain"),
        "fold": fold
    })
    feature_importance_df = pd.concat([feature_importance_df, fold_imp])

    del clf, train_set, val_set, X_tr, X_val, y_tr, y_val
    gc.collect()

# outputs
print("\n---- OOF Scores ----")
print("OOF ROC AUC:", roc_auc_score(y, oof_preds))
print("OOF PR AUC :", average_precision_score(y, oof_preds))

# save submission
test['isFraud'] = test_preds
test[[IDCOL, 'isFraud']].to_csv(OUT_PATH, index=False)
print(f"Saved submission to {OUT_PATH}")

# feature importance
fi = feature_importance_df.groupby("feature")['importance'].mean().sort_values(ascending=False)
print("\nTop 30 important features:")
print(fi.head(30))


