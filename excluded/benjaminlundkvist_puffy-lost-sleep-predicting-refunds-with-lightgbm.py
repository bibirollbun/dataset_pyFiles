# --- Setup ---
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

pd.set_option('display.max_columns', 200)
plt.rcParams['figure.figsize'] = (10, 5)
sns.set(style="whitegrid")


data_dir = Path('/kaggle/input/the-puffy-lost-sleepchallenge')

train = pd.read_parquet(data_dir / 'train.parquet')
test = pd.read_parquet(data_dir / 'test.parquet')
events = pd.read_parquet(data_dir / 'events.parquet')
sample_submission = pd.read_csv(data_dir / 'sample_submission.csv')

print("Shapes:", "train", train.shape, "test", test.shape, "events", events.shape)

target_col = 'refunded'

# Quick look at target distribution
sns.countplot(x=target_col, data=train)
plt.title("Distribution of Refunded Orders")
plt.show()


# --- Extract order-level features ---
def extract_order_features(df):
    order_value, n_items, unique_products = [], [], []
    for s in df['line_items'].fillna('[]').astype(str):
        try:
            items = json.loads(s)
        except Exception:
            try:
                items = eval(s)
            except Exception:
                items = []
        total_cents, qty, prods = 0, 0, set()
        for it in items:
            if isinstance(it, dict):
                price = it.get('price', 0) or 0
                quantity = it.get('quantity', 1) or 1
                item_id = it.get('item_id') or it.get('product_id') or None
            else:
                price, quantity, item_id = 0, 1, None
            try:
                total_cents += int(price) * int(quantity)
            except Exception:
                try:
                    total_cents += float(price) * int(quantity)
                except Exception:
                    pass
            qty += int(quantity)
            if item_id is not None:
                prods.add(item_id)
        order_value.append(total_cents / 100.0)
        n_items.append(qty)
        unique_products.append(len(prods))
    df = df.copy()
    df['order_value'] = order_value
    df['n_items'] = n_items
    df['n_unique_products'] = unique_products
    return df

train = extract_order_features(train)
test = extract_order_features(test)

# --- Aggregate client events ---
events = events.copy()
events['event_timestamp'] = pd.to_datetime(events['event_timestamp'], utc=True)

agg_funcs = {
    'event_id': 'count',
    'event_name': pd.Series.nunique,
    'page_url': pd.Series.nunique,
    'event_timestamp': ['min', 'max']
}

client_agg = events.groupby('client_id').agg(agg_funcs)
client_agg.columns = ['total_events','unique_event_types','distinct_pages','first_event_ts','last_event_ts']
client_agg = client_agg.reset_index()
client_agg['session_span_seconds'] = (client_agg['last_event_ts'] - client_agg['first_event_ts']).dt.total_seconds().fillna(0)

# Merge client features with train/test
train = train.merge(client_agg, on='client_id', how='left')
test = test.merge(client_agg, on='client_id', how='left')

train = train.fillna(0)
test = test.fillna(0)

# --- Time-lag features ---
def add_time_lag_features(df):
    if 'last_event_ts' in df.columns and 'order_timestamp' in df.columns:
        df['order_timestamp'] = pd.to_datetime(df['order_timestamp'], utc=True, errors='coerce')
        df['last_event_ts'] = pd.to_datetime(df['last_event_ts'], utc=True, errors='coerce')
        df['time_since_last_event_hours'] = (
            (df['order_timestamp'] - df['last_event_ts']).dt.total_seconds() / 3600.0
        ).fillna(9999).clip(lower=-1000, upper=9999)
    else:
        df['time_since_last_event_hours'] = 9999
    df['events_per_hour'] = df['total_events'] / ((df['session_span_seconds'] / 3600.0).replace(0, np.nan))
    df['events_per_hour'] = df['events_per_hour'].fillna(0)
    return df

train = add_time_lag_features(train)
test = add_time_lag_features(test)


sns.histplot(train['order_value'], bins=50, kde=True)
plt.title("Distribution of Order Value ($)")
plt.show()

sns.histplot(train['n_items'], bins=30, kde=False)
plt.title("Number of Items per Order")
plt.show()

sns.histplot(train['time_since_last_event_hours'], bins=50, kde=True)
plt.title("Time Since Last Event (hours)")
plt.show()


exclude_cols = {'order_id','client_id','line_items','event_timestamp','event_data','page_url','user_agent',
                 'first_event_ts','last_event_ts','order_timestamp'}
features = [c for c in train.columns if c not in exclude_cols and c != target_col]


X, y = train[features].copy(), train[target_col].astype(int).values
X_test = test[features].copy()
test_order_ids = test['order_id'].values

# Handle missing/inf values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        le = LabelEncoder()
        vals = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0)
        le.fit(vals)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# LightGBM model setup
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

# Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

# Train with K-Fold CV
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nTraining fold {fold}/{n_splits} ...")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    oof[val_idx] = lgb_model.predict_proba(X_val)[:,1]
    test_pred += lgb_model.predict_proba(X_test)[:,1] / n_splits

# Out-of-fold AUC
oof_auc = roc_auc_score(y, oof)
print(f"\nOOF ROC AUC: {oof_auc:.5f}")


submission = pd.DataFrame({
    'order_id': test['order_id'],
    'refunded': test_pred.clip(0,1)
})

submission.to_csv('submission.csv', index=False)
print(f"Saved submission.csv with {len(submission)} rows using LightGBM")
submission.head()

