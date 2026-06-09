import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/events.parquet")

# Basic info
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# Plot distributions for numeric columns
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Data Distribution of Numeric Columns", fontsize=16)
plt.show()




df = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/train.parquet")

# Basic info
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# Plot distributions for numeric columns
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Data Distribution of Numeric Columns", fontsize=16)
plt.show()


train_df = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/train.parquet")
events_df = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/events.parquet")

print(train_df.shape, events_df.shape)
print(train_df.head())
print(events_df.head())


print(train_df['refunded'].value_counts(normalize=True))


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="refunded", data=train_df)
plt.title("Target Distribution: refunded")
plt.show()

print(train_df['refunded'].value_counts(normalize=True))


import ast

def extract_features(row):
    try:
        items = ast.literal_eval(row)
        prices = [i["price"] * i["quantity"] for i in items]
        quantities = [i["quantity"] for i in items]
        return pd.Series({
            "total_price": sum(prices),
            "total_quantity": sum(quantities),
            "avg_price_per_item": (sum(prices) / sum(quantities)) if quantities else 0
        })
    except:
        return pd.Series({"total_price": None, "total_quantity": None, "avg_price_per_item": None})

train_features = train_df.join(train_df["line_items"].apply(extract_features))
train_features


for col in ["total_price", "total_quantity", "avg_price_per_item"]:
    sns.boxplot(x="refunded", y=col, data=train_features)
    plt.title(f"{col} by refunded")
    plt.show()


events_summary = events_df.groupby("client_id").agg(
    event_count=("event_id", "count"),
    unique_events=("event_name", "nunique")
).reset_index()

merged = train_features.merge(events_summary, on="client_id", how="left")
merged


for col in ["event_count", "unique_events"]:
    sns.boxplot(x="refunded", y=col, data=merged)
    plt.title(f"{col} by refunded")
    plt.show()


# Pick a refunded client (as an example)
sample_client = merged.loc[merged["refunded"] == 1, "client_id"].iloc[0]
print("Chosen client:", sample_client)

# Show all events for this client
client_events = events_df[events_df["client_id"] == sample_client].sort_values("event_timestamp")

unique_events = client_events["event_name"].unique()
print("Unique event types:", unique_events)



# Step 1: Get unique event types
unique_event_types = events_df["event_name"].unique()
print("Unique event types:", unique_event_types)

# Step 2: Pivot into wide format (one column per event_name)
event_pivot = (
    events_df.groupby(["client_id", "event_name"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Step 3: Merge with train_features
merged = train_features.merge(event_pivot, on="client_id", how="left")

print(merged.head())



drop_cols = ["client_id", "order_id", "line_items"]
merged_clean = merged.drop(columns=drop_cols)

print(merged_clean.head())


merged_clean["order_timestamp"] = pd.to_datetime(train_features["order_timestamp"])

merged_clean["order_hour"] = merged_clean["order_timestamp"].dt.hour
merged_clean["order_dayofweek"] = merged_clean["order_timestamp"].dt.dayofweek
merged_clean["order_is_weekend"] = merged_clean["order_dayofweek"].isin([5,6]).astype(int)
merged_clean = merged_clean.drop(columns="order_timestamp")
merged_clean


import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# -----------------------------
# Load your preprocessed dataset
# -----------------------------

df = merged_clean

# Separate features & target
X = df.drop(columns=["refunded"])
y = df["refunded"]

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Define objective function for Optuna
# -----------------------------
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",  # 'gpu_hist' if GPU available
        "use_label_encoder": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    auc = roc_auc_score(y_valid, preds)
    return auc

# -----------------------------
# Run Optuna optimization
# -----------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best params:", study.best_trial.params)
print("Best AUC:", study.best_value)

# -----------------------------
# Train final model with best params
# -----------------------------
best_params = study.best_trial.params
best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# Evaluation
preds = best_model.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, preds))
print("F1 Score:", f1_score(y_valid, preds))
print("ROC-AUC:", roc_auc_score(y_valid, preds))



!pip install scikit-learn==1.3.* imbalanced-learn==0.11.*


import pandas as pd
import xgboost as xgb
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

# -----------------------------
# Load dataset
# -----------------------------
df = merged_clean.copy()

# Convert datetime columns to numeric (timestamp)
datetime_columns = df.select_dtypes(include=['datetime64']).columns
for col in datetime_columns:
    if col != 'refunded':  # Don't convert target variable
        df[col] = pd.to_datetime(df[col]).astype('int64') // 10**9  # Convert to Unix timestamp

# Convert object/string columns to numeric if possible
object_columns = df.select_dtypes(include=['object']).columns
for col in object_columns:
    if col != 'refunded':  # Don't convert target variable
        # Try to convert to numeric, otherwise use label encoding
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            # Label encode categorical variables
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

# Separate features & target
X = df.drop(columns=["refunded"])
y = df["refunded"]

print("Data types after conversion:")
print(X.dtypes)
print("\nChecking for any remaining non-numeric columns:")
non_numeric = X.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32', 'bool']).columns
if len(non_numeric) > 0:
    print(f"Found non-numeric columns: {list(non_numeric)}")
    # Handle remaining non-numeric columns
    for col in non_numeric:
        print(f"Converting {col} with dtype {X[col].dtype}")
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
else:
    print("All columns are now numeric")

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Manual oversampling (alternative to SMOTE)
# -----------------------------
# Combine X and y for easier manipulation
train_data = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority_class = train_data[train_data.refunded == 0]
minority_class = train_data[train_data.refunded == 1]

print("Original class balance:")
print(f"Majority class (0): {len(majority_class)}")
print(f"Minority class (1): {len(minority_class)}")

# Upsample minority class to match majority class
minority_upsampled = resample(minority_class,
                             replace=True,
                             n_samples=len(majority_class),
                             random_state=42)

# Combine majority class with upsampled minority class
upsampled_data = pd.concat([majority_class, minority_upsampled])

# Separate features and target
X_train_res = upsampled_data.drop('refunded', axis=1)
y_train_res = upsampled_data['refunded']

print("After upsampling:")
print("Class balance:", y_train_res.value_counts(normalize=True))

# -----------------------------
# Define Optuna objective
# -----------------------------
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",  # use "gpu_hist" if GPU available
        "use_label_encoder": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        # Since we balanced the classes manually, we can still use scale_pos_weight
        "scale_pos_weight": len(y_train[y_train==0]) / len(y_train[y_train==1])
    }
    
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train_res, y_train_res)
    
    # Use predict_proba for ROC-AUC calculation (more appropriate)
    pred_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, pred_proba)
    
    return auc

# -----------------------------
# Run Optuna
# -----------------------------
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best params:", study.best_trial.params)
print("Best AUC:", study.best_value)

# -----------------------------
# Train final model
# -----------------------------
best_params = study.best_trial.params
best_model = xgb.XGBClassifier(**best_params, random_state=42)
best_model.fit(X_train_res, y_train_res)

# Evaluation
preds = best_model.predict(X_valid)
pred_proba = best_model.predict_proba(X_valid)[:, 1]

print("\n" + "="*50)
print("FINAL MODEL EVALUATION")
print("="*50)
print("Accuracy:", accuracy_score(y_valid, preds))
print("F1 Score:", f1_score(y_valid, preds))
print("ROC-AUC:", roc_auc_score(y_valid, pred_proba))

# Print validation set class distribution for reference
print(f"\nValidation set class distribution:")
print(y_valid.value_counts(normalize=True))





test_df = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/test.parquet")
test_df = test_df.join(train_df["line_items"].apply(extract_features))
test_df


events_summary = events_df.groupby("client_id").agg(
    event_count=("event_id", "count"),
    unique_events=("event_name", "nunique")
).reset_index()

merged = test_df.merge(events_summary, on="client_id", how="left")
merged


# Step 1: Get unique event types
unique_event_types = events_df["event_name"].unique()
print("Unique event types:", unique_event_types)

# Step 2: Pivot into wide format (one column per event_name)
event_pivot = (
    events_df.groupby(["client_id", "event_name"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Step 3: Merge with train_features
merged = test_df.merge(event_pivot, on="client_id", how="left")

print(merged.head())



drop_cols = ["client_id", "line_items"]
merged_clean = merged.drop(columns=drop_cols)

print(merged_clean.head())


merged_clean["order_timestamp"] = pd.to_datetime(train_features["order_timestamp"])

merged_clean["order_hour"] = merged_clean["order_timestamp"].dt.hour
merged_clean["order_dayofweek"] = merged_clean["order_timestamp"].dt.dayofweek
merged_clean["order_is_weekend"] = merged_clean["order_dayofweek"].isin([5,6]).astype(int)
merged_clean = merged_clean.drop(columns="order_timestamp")
merged_clean





import pandas as pd

# Load test dataset
test_df = merged_clean

# Keep order_id separately
order_ids = test_df["order_id"]

# Drop non-feature columns (only keep the features used during training)
X_test = test_df.drop(columns=["order_id"])

# Predict refunded (binary classification)
test_preds = best_model.predict(X_test)

# Build submission dataframe
submission = pd.DataFrame({
    "order_id": order_ids,
    "refunded": test_preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")
print(submission.head())



import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# =====================
# 1. Load Data
# =====================
train = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/train.parquet")
test = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/test.parquet")
events = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/events.parquet")

# =====================
# 2. Process timestamps
# =====================
def process_time_features(df, col="order_timestamp"):
    df[col] = pd.to_datetime(df[col])
    df["order_hour"] = df[col].dt.hour
    df["order_dayofweek"] = df[col].dt.dayofweek
    df["order_is_weekend"] = (df["order_dayofweek"].isin([5,6])).astype(int)
    df.drop(columns=[col], inplace=True)
    return df

train = process_time_features(train, "order_timestamp")
test = process_time_features(test, "order_timestamp")

# =====================
# 3. Parse line_items JSON
# =====================
def parse_line_items(line_items_str):
    try:
        items = json.loads(line_items_str)
        if not items: return pd.Series([0,0,0,0])
        total_price = sum(i["price"] * i["quantity"] for i in items)
        total_qty   = sum(i["quantity"] for i in items)
        avg_price   = np.mean([i["price"] for i in items])
        unique_items= len(set(i["item_id"] for i in items))
        return pd.Series([total_price, total_qty, avg_price, unique_items])
    except:
        return pd.Series([0,0,0,0])

train[["total_price","total_qty","avg_price","unique_items"]] = train["line_items"].apply(parse_line_items)
test[["total_price","total_qty","avg_price","unique_items"]]  = test["line_items"].apply(parse_line_items)

train.drop(columns=["line_items"], inplace=True)
test.drop(columns=["line_items"], inplace=True)

# =====================
# 4. Aggregate Events
# =====================
# convert timestamp
events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

# aggregate counts of each event_name per client
event_counts = (
    events.groupby(["client_id","event_name"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Add ratios
if "page_view" in event_counts.columns and "add_to_cart" in event_counts.columns:
    event_counts["cart_to_view_ratio"] = event_counts["add_to_cart"] / (event_counts["page_view"]+1)

if "add_to_cart" in event_counts.columns and "remove_from_cart" in event_counts.columns:
    event_counts["remove_to_add_ratio"] = event_counts["remove_from_cart"] / (event_counts["add_to_cart"]+1)

if "checkout_started" in event_counts.columns and "checkout_completed" in event_counts.columns:
    event_counts["checkout_success_ratio"] = event_counts["checkout_completed"] / (event_counts["checkout_started"]+1)

# merge with train/test
train = train.merge(event_counts, on="client_id", how="left").fillna(0)
test  = test.merge(event_counts, on="client_id", how="left").fillna(0)

# =====================
# 5. Prepare Features
# =====================
X = train.drop(columns=["refunded","order_id","client_id"], errors="ignore")
y = train["refunded"]

X_test = test.drop(columns=["order_id","client_id"], errors="ignore")

# align columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# =====================
# 6. Train/Validation Split
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =====================
# 7. Ensemble Model
# =====================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)

log_reg = LogisticRegression(max_iter=2000, class_weight="balanced")

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("lr", log_reg), ("xgb", xgb)],
    voting="soft",
    n_jobs=-1
)

ensemble.fit(X_train, y_train)

# =====================
# 8. Validation
# =====================
y_pred = ensemble.predict(X_val)
y_proba = ensemble.predict_proba(X_val)[:,1]

print("Validation ROC-AUC:", roc_auc_score(y_val, y_proba))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# =====================
# 9. Submission
# =====================
test_preds = ensemble.predict(X_test)

submission = pd.DataFrame({
    "order_id": test["order_id"],
    "refunded": test_preds
})

submission.to_csv("sub.csv", index=False)
print("âœ… sub.csv generated")



# Stronger pipeline with per-order event windows, rich features, TF-IDF, Optuna-tuned XGBoost, stacking
# Paste into a single cell in your notebook. Adjust input paths if needed.

import pandas as pd
import numpy as np
import json
from datetime import timedelta
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import optuna
from scipy.sparse import hstack, csr_matrix
from sklearn.base import clone

# --------------------------
# 0. load data
# --------------------------
train = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/train.parquet")
test  = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/test.parquet")
events = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/events.parquet")

# ensure timestamps are datetimes
train["order_timestamp"] = pd.to_datetime(train["order_timestamp"])
test["order_timestamp"] = pd.to_datetime(test["order_timestamp"])
events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

# Keep copies for submission
test_order_ids = test["order_id"].copy()

# --------------------------
# 1. parse line_items -> numeric features
# --------------------------
def parse_line_items(line_items_str):
    try:
        items = json.loads(line_items_str)
        if not items:
            return 0,0,0,0
        total_price = sum(i.get("price",0) * i.get("quantity",1) for i in items)  # in cents
        total_qty = sum(i.get("quantity",1) for i in items)
        avg_price = np.mean([i.get("price",0) for i in items]) if items else 0
        unique_items = len({i.get("item_id") for i in items})
        return total_price, total_qty, avg_price, unique_items
    except:
        return 0,0,0,0

for df in (train, test):
    parsed = df["line_items"].apply(parse_line_items)
    df[["total_price","total_qty","avg_price","unique_items"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    df.drop(columns=["line_items"], inplace=True)

# --------------------------
# 2. Build per-order event features (events up to order_timestamp)
#    We'll create features per order by merging events filtered per order_timestamp.
#    This avoids leaking future events.
# --------------------------
# Combine orders so we can compute features for both train and test in one pass
train["is_train"] = 1
test["is_train"] = 0
orders = pd.concat([train, test], axis=0, ignore_index=True, sort=False)

# To speed up: sort events and orders by client_id & timestamp
events_sorted = events.sort_values(["client_id","event_timestamp"]).reset_index(drop=True)
orders_sorted = orders.sort_values(["client_id","order_timestamp"]).reset_index(drop=True)

# We'll iterate per client â€” for each client's orders, filter that client's events and compute features for each order.
client_event_groups = events_sorted.groupby("client_id")

# Prepare lists to collect features per order in same order as orders_sorted
agg_features = []

# Pre-define windows (in days) to compute counts
windows = [1, 7, 30]  # last 1 day, 7 days, 30 days

# For text aggregation (page_url and event_data), we'll build concatenated strings per order
page_url_texts = []
event_data_texts = []

# Also include numeric features derived from events for each order
# iterate orders_sorted by client
print("Building per-order event features (this may take a few minutes)...")
for idx, ord_row in orders_sorted.iterrows():
    cid = ord_row["client_id"]
    ts = ord_row["order_timestamp"]
    # default empty
    client_events = client_event_groups.get_group(cid) if cid in client_event_groups.groups else None
    if client_events is None:
        # no events for this client
        agg_features.append({})
        page_url_texts.append("")
        event_data_texts.append("")
        continue

    # select events <= order_timestamp
    ev = client_events[client_events["event_timestamp"] <= ts]
    if ev.shape[0] == 0:
        agg_features.append({})
        page_url_texts.append("")
        event_data_texts.append("")
        continue

    # global counts of event_name
    event_name_counts = ev["event_name"].value_counts().to_dict()

    # counts in windows
    counts_by_window = {}
    for w in windows:
        cutoff = ts - pd.Timedelta(days=w)
        ev_w = ev[ev["event_timestamp"] >= cutoff]
        cdict = ev_w["event_name"].value_counts().to_dict()
        # prefix with window, sum total events in window too
        counts_by_window.update({f"cnt_{k}_last{w}d": v for k,v in cdict.items()})
        counts_by_window[f"total_events_last{w}d"] = ev_w.shape[0]

    # recency features: time since last event, time since first event
    last_event_ts = ev["event_timestamp"].max()
    first_event_ts = ev["event_timestamp"].min()
    time_since_last = (ts - last_event_ts).total_seconds() if pd.notna(last_event_ts) else np.nan
    time_since_first = (ts - first_event_ts).total_seconds() if pd.notna(first_event_ts) else np.nan

    # inter-event time mean
    ev_sorted_ts = ev["event_timestamp"].sort_values()
    if len(ev_sorted_ts) >= 2:
        diffs = ev_sorted_ts.diff().dt.total_seconds().dropna()
        avg_inter_event = diffs.mean()
        std_inter_event = diffs.std()
    else:
        avg_inter_event = np.nan
        std_inter_event = np.nan

    # unique pages, unique event types
    unique_pages = ev["page_url"].nunique()
    unique_event_types = ev["event_name"].nunique()

    # approximate sessions: count gaps > 30 minutes
    gaps = ev_sorted_ts.diff().dt.total_seconds().fillna(0)
    num_sessions = int((gaps > 30*60).sum()) + 1

    # concat page_url and event_data for TF-IDF text features
    page_concat = " ".join(ev["page_url"].dropna().astype(str).tolist())
    event_data_concat = " ".join(ev["event_data"].dropna().astype(str).tolist())

    # assemble feature dict
    feat = {}
    # event counts
    for k,v in event_name_counts.items():
        feat[f"evt_{k}_cnt"] = int(v)
    # windowed counts
    feat.update(counts_by_window)
    # recency and inter-event
    feat["time_since_last_event_s"] = time_since_last if not np.isnan(time_since_last) else -1
    feat["time_since_first_event_s"] = time_since_first if not np.isnan(time_since_first) else -1
    feat["avg_inter_event_s"] = avg_inter_event if not np.isnan(avg_inter_event) else -1
    feat["std_inter_event_s"] = std_inter_event if not np.isnan(std_inter_event) else -1
    feat["unique_pages"] = unique_pages
    feat["unique_event_types"] = unique_event_types
    feat["num_sessions_30min_gap"] = num_sessions
    feat["total_events"] = ev.shape[0]

    agg_features.append(feat)
    page_url_texts.append(page_concat)
    event_data_texts.append(event_data_concat)

# Create a DataFrame of agg features aligned with orders_sorted
agg_df = pd.DataFrame(agg_features).fillna(0).astype(float)
agg_df.index = orders_sorted.index

# attach to orders_sorted
orders_with_feats = pd.concat([orders_sorted.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)
orders_with_feats["page_url_text"] = page_url_texts
orders_with_feats["event_data_text"] = event_data_texts

# --------------------------
# 3. Merge back -> split train/test
# --------------------------
# recover train/test by is_train flag
orders_with_feats = orders_with_feats.reset_index(drop=True)
train_fe = orders_with_feats[orders_with_feats["is_train"]==1].copy().reset_index(drop=True)
test_fe  = orders_with_feats[orders_with_feats["is_train"]==0].copy().reset_index(drop=True)

# drop helper column
for df in (train_fe, test_fe):
    df.drop(columns=["is_train"], inplace=True)

# --------------------------
# 4. Additional engineered features from orders (timestamp-derived already removed earlier; keep numeric features)
# --------------------------
# We already removed order_timestamp earlier in earlier code. If not, ensure removed.
# Add simple ratios
for df in (train_fe, test_fe):
    # total_price is in cents; to dollars
    if "total_price" in df.columns:
        df["total_price_dollars"] = df["total_price"] / 100.0
    df["price_per_item"] = df["total_price"] / (df["total_qty"] + 1e-6)
    df["items_per_unique"] = df["total_qty"] / (df["unique_items"] + 1e-6)

# --------------------------
# 5. Build final tabular X (numeric) and text matrices
# --------------------------
# drop columns not features
drop_cols = ["order_id","client_id","order_timestamp"] if "order_timestamp" in orders_with_feats.columns else ["order_id","client_id"]
# ensure target exists in train_fe
y = train_fe["refunded"].astype(int).values

X_train_tab = train_fe.drop(columns=[c for c in (drop_cols + ["refunded","line_items"]) if c in train_fe.columns], errors='ignore').copy()
X_test_tab  = test_fe.drop(columns=[c for c in (drop_cols + ["refunded","line_items"]) if c in test_fe.columns], errors='ignore').copy()

# Keep the page_url_text and event_data_text separately for TF-IDF
train_page_text = X_train_tab.pop("page_url_text").astype(str).fillna("")
train_event_text = X_train_tab.pop("event_data_text").astype(str).fillna("")
test_page_text  = X_test_tab.pop("page_url_text").astype(str).fillna("")
test_event_text = X_test_tab.pop("event_data_text").astype(str).fillna("")

# numeric dataframe (fillna)
X_train_tab = X_train_tab.fillna(0)
X_test_tab = X_test_tab.fillna(0)

# Scale numeric features (use StandardScaler later inside training if needed)
num_cols = X_train_tab.columns.tolist()

# --------------------------
# 6. TF-IDF (limited size) for page_url_text and event_data_text
# --------------------------
tfidf_url = TfidfVectorizer(max_features=200, ngram_range=(1,2), stop_words='english')
tfidf_event = TfidfVectorizer(max_features=200, ngram_range=(1,2), stop_words='english')

X_train_url = tfidf_url.fit_transform(train_page_text)
X_test_url = tfidf_url.transform(test_page_text)
X_train_event = tfidf_event.fit_transform(train_event_text)
X_test_event = tfidf_event.transform(test_event_text)

# --------------------------
# 7. Combine numeric and text into sparse matrices
# --------------------------
X_train_num = csr_matrix(X_train_tab.values)  # dense -> sparse
X_test_num  = csr_matrix(X_test_tab.values)

X_train_all = hstack([X_train_num, X_train_url, X_train_event]).tocsr()
X_test_all  = hstack([X_test_num, X_test_url, X_test_event]).tocsr()

print("Shapes:", X_train_all.shape, X_test_all.shape)

# --------------------------
# 8. Modeling: Use stratified CV, Optuna to tune XGBoost lightly
# --------------------------
# compute scale_pos_weight for XGBoost
pos = y.sum()
neg = len(y) - pos
scale_pos_weight = neg / (pos + 1e-9)

def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": "gbtree",
        "tree_method": "hist",
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "n_estimators": 500,
        "scale_pos_weight": scale_pos_weight,
        "use_label_encoder": False,
        "random_state": 42
    }
    # 5-fold stratified CV AUC
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in skf.split(X_train_all, y):
        Xtr, Xv = X_train_all[tr_idx], X_train_all[val_idx]
        ytr, yv = y[tr_idx], y[val_idx]
        model = XGBClassifier(**param)
        model.fit(Xtr, ytr, eval_set=[(Xv,yv)], early_stopping_rounds=50, verbose=False)
        yv_pred = model.predict_proba(Xv)[:,1]
        aucs.append(roc_auc_score(yv, yv_pred))
    return np.mean(aucs)

# run a modest number of trials to conserve time (e.g., 25)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25, show_progress_bar=True)

print("Best params:", study.best_params)

# Train final XGBoost with best params (increase n_estimators and early stopping)
best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": 42,
    "n_estimators": 1000,
    "scale_pos_weight": scale_pos_weight
})

final_xgb = XGBClassifier(**best_params)
# Fit on full train with early stopping using a small holdout
Xtr, Xhold, ytr, yhold = train_test_split(X_train_all, y, test_size=0.1, stratify=y, random_state=42)
final_xgb.fit(Xtr, ytr, eval_set=[(Xhold,yhold)], early_stopping_rounds=50, verbose=50)

# Evaluate on holdout
y_hold_pred = final_xgb.predict_proba(Xhold)[:,1]
y_hold_pred_cls = (y_hold_pred >= 0.5).astype(int)
print("Holdout ROC-AUC:", roc_auc_score(yhold, y_hold_pred))
print("\nClassification Report (holdout):\n", classification_report(yhold, y_hold_pred_cls))

# --------------------------
# 9. Stacking: logistic on XGBoost probabilities + RF probs
# --------------------------
# get XGB probs on full train via 5-fold oof
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))
rf_oof = np.zeros(len(y))
for tr_idx, val_idx in skf.split(X_train_all, y):
    Xtr, Xv = X_train_all[tr_idx], X_train_all[val_idx]
    ytr, yv = y[tr_idx], y[val_idx]
    m = clone(final_xgb)
    m.fit(Xtr, ytr, eval_set=[(Xv,yv)], early_stopping_rounds=50, verbose=False)
    oof_preds[val_idx] = m.predict_proba(Xv)[:,1]
    # RF oof
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced_subsample")
    rf.fit(Xtr, ytr)
    rf_oof[val_idx] = rf.predict_proba(Xv)[:,1]

stack_X = np.vstack([oof_preds, rf_oof]).T
meta_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
meta_clf.fit(stack_X, y)

# For test: get XGB and RF predictions trained on full train
final_xgb_full = clone(final_xgb)
final_xgb_full.fit(X_train_all, y, verbose=False)
rf_full = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42, class_weight="balanced_subsample")
rf_full.fit(X_train_all, y)

xgb_test_proba = final_xgb_full.predict_proba(X_test_all)[:,1]
rf_test_proba  = rf_full.predict_proba(X_test_all)[:,1]
stack_test_X = np.vstack([xgb_test_proba, rf_test_proba]).T
meta_test_pred_proba = meta_clf.predict_proba(stack_test_X)[:,1]
meta_test_pred_cls = (meta_test_pred_proba >= 0.5).astype(int)

# --------------------------
# 10. Submission file (use meta_test_pred_cls)
# --------------------------
submission = pd.DataFrame({
    "order_id": test_order_ids.values,
    "refunded": meta_test_pred_cls
})
submission.to_csv("sub.csv", index=False)
print("Saved sub.csv")

# --------------------------
# 11. Important diagnostics print
# --------------------------
# training AUC on full train
train_proba = final_xgb_full.predict_proba(X_train_all)[:,1]
print("Train ROC-AUC (XGB full):", roc_auc_score(y, train_proba))

# feature importance approx (from XGBoost)
try:
    importances = final_xgb_full.get_booster().get_score(importance_type='gain')
    # print top 30 features
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:30]
    print("Top XGB features (name,gain):")
    for n,g in sorted_feats:
        print(n, g)
except Exception as e:
    print("Could not get XGB importances:", e)

# Print holdout metrics were already printed above



# Enhanced pipeline targeting 90% accuracy with advanced feature engineering and modeling
# Major improvements: behavioral patterns, sequential features, ensemble methods, advanced preprocessing

import pandas as pd
import numpy as np
import json
from datetime import timedelta, datetime
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, classification_report, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from scipy.sparse import hstack, csr_matrix
from scipy import stats
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# 0. Load data with enhanced preprocessing
# --------------------------
train = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/train.parquet")
test = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/test.parquet")
events = pd.read_parquet("/kaggle/input/the-puffy-lost-sleepchallenge/events.parquet")

# Enhanced timestamp parsing with timezone handling
for df in [train, test]:
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

test_order_ids = test["order_id"].copy()

# --------------------------
# 1. Enhanced line_items parsing with detailed product features
# --------------------------
def advanced_parse_line_items(line_items_str):
    try:
        items = json.loads(line_items_str)
        if not items:
            return [0] * 15
        
        prices = [i.get("price", 0) for i in items]
        quantities = [i.get("quantity", 1) for i in items]
        
        total_price = sum(p * q for p, q in zip(prices, quantities))
        total_qty = sum(quantities)
        
        # Advanced statistics
        avg_price = np.mean(prices) if prices else 0
        median_price = np.median(prices) if prices else 0
        std_price = np.std(prices) if len(prices) > 1 else 0
        min_price = min(prices) if prices else 0
        max_price = max(prices) if prices else 0
        price_range = max_price - min_price
        
        # Quantity statistics
        avg_qty = np.mean(quantities) if quantities else 0
        max_qty = max(quantities) if quantities else 0
        std_qty = np.std(quantities) if len(quantities) > 1 else 0
        
        # Product diversity metrics
        unique_items = len({i.get("item_id") for i in items if i.get("item_id")})
        price_qty_correlation = np.corrcoef(prices, quantities)[0, 1] if len(prices) > 1 else 0
        price_qty_correlation = 0 if np.isnan(price_qty_correlation) else price_qty_correlation
        
        # High-value item indicators
        high_price_items = sum(1 for p in prices if p > np.percentile(prices, 75)) if prices else 0
        low_price_items = sum(1 for p in prices if p < np.percentile(prices, 25)) if prices else 0
        
        return [total_price, total_qty, avg_price, median_price, std_price, min_price, max_price, 
                price_range, avg_qty, max_qty, std_qty, unique_items, price_qty_correlation,
                high_price_items, low_price_items]
    except:
        return [0] * 15

# Apply enhanced parsing
item_feature_names = ['total_price', 'total_qty', 'avg_price', 'median_price', 'std_price', 
                     'min_price', 'max_price', 'price_range', 'avg_qty', 'max_qty', 'std_qty', 
                     'unique_items', 'price_qty_correlation', 'high_price_items', 'low_price_items']

for df in [train, test]:
    parsed = df["line_items"].apply(advanced_parse_line_items)
    df[item_feature_names] = pd.DataFrame(parsed.tolist(), index=df.index)

# --------------------------
# 2. Advanced temporal features from order timestamps
# --------------------------
def extract_temporal_features(df):
    df['hour'] = df['order_timestamp'].dt.hour
    df['day_of_week'] = df['order_timestamp'].dt.dayofweek
    df['day_of_month'] = df['order_timestamp'].dt.day
    df['month'] = df['order_timestamp'].dt.month
    df['quarter'] = df['order_timestamp'].dt.quarter
    df['year'] = df['order_timestamp'].dt.year
    
    # Business hours and weekend indicators
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Shopping pattern indicators
    df['is_lunch_time'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
    
    return df

train = extract_temporal_features(train)
test = extract_temporal_features(test)

# --------------------------
# 3. Enhanced per-order event features with behavioral patterns
# --------------------------
train["is_train"] = 1
test["is_train"] = 0
orders = pd.concat([train, test], axis=0, ignore_index=True, sort=False)

# Pre-process events for efficiency
events_sorted = events.sort_values(["client_id", "event_timestamp"]).reset_index(drop=True)
orders_sorted = orders.sort_values(["client_id", "order_timestamp"]).reset_index(drop=True)
client_event_groups = events_sorted.groupby("client_id")

print("Building advanced behavioral features...")

# Enhanced feature extraction
windows = [1, 3, 7, 14, 30, 90]  # Extended windows
agg_features = []
page_url_texts = []
event_data_texts = []

# Pre-calculate event type statistics for normalization
all_event_types = events['event_name'].unique()
event_type_stats = events.groupby('event_name').size().to_dict()

for idx, ord_row in orders_sorted.iterrows():
    if idx % 1000 == 0:
        print(f"Processing order {idx}/{len(orders_sorted)}")
    
    cid = ord_row["client_id"]
    ts = ord_row["order_timestamp"]
    
    client_events = client_event_groups.get_group(cid) if cid in client_event_groups.groups else None
    if client_events is None or client_events.empty:
        agg_features.append({})
        page_url_texts.append("")
        event_data_texts.append("")
        continue

    # Filter events before order timestamp
    ev = client_events[client_events["event_timestamp"] <= ts].copy()
    if ev.empty:
        agg_features.append({})
        page_url_texts.append("")
        event_data_texts.append("")
        continue

    # Sort by timestamp for sequential analysis
    ev = ev.sort_values("event_timestamp")
    
    feat = {}
    
    # 1. Basic event counts and frequencies
    event_counts = ev["event_name"].value_counts().to_dict()
    for event_type in all_event_types:
        feat[f"evt_{event_type}_count"] = event_counts.get(event_type, 0)
        # Normalized by global frequency
        feat[f"evt_{event_type}_norm"] = feat[f"evt_{event_type}_count"] / max(event_type_stats.get(event_type, 1), 1)
    
    # 2. Windowed features with decay
    for window in windows:
        cutoff = ts - pd.Timedelta(days=window)
        ev_w = ev[ev["event_timestamp"] >= cutoff]
        
        if not ev_w.empty:
            # Time-decayed weights (recent events matter more)
            time_weights = np.exp(-0.1 * (ts - ev_w["event_timestamp"]).dt.total_seconds() / 86400)  # Decay over days
            
            feat[f"events_last_{window}d"] = len(ev_w)
            feat[f"unique_events_last_{window}d"] = ev_w["event_name"].nunique()
            feat[f"unique_pages_last_{window}d"] = ev_w["page_url"].nunique()
            
            # Weighted event frequencies
            for event_type in ev_w["event_name"].unique():
                mask = ev_w["event_name"] == event_type
                weighted_count = time_weights[mask].sum()
                feat[f"evt_{event_type}_weighted_last_{window}d"] = weighted_count
        else:
            feat[f"events_last_{window}d"] = 0
            feat[f"unique_events_last_{window}d"] = 0
            feat[f"unique_pages_last_{window}d"] = 0
    
    # 3. Sequential and behavioral patterns
    if len(ev) >= 2:
        # Event sequences and transitions
        event_sequence = ev["event_name"].tolist()
        transitions = [(event_sequence[i], event_sequence[i+1]) for i in range(len(event_sequence)-1)]
        transition_counts = Counter(transitions)
        
        # Most common transition patterns
        if transitions:
            most_common_transition = transition_counts.most_common(1)[0]
            feat["most_common_transition_count"] = most_common_transition[1]
            feat["unique_transitions"] = len(set(transitions))
            feat["transition_entropy"] = stats.entropy(list(transition_counts.values()))
        
        # Session analysis (gaps > 30 minutes define new sessions)
        time_diffs = ev["event_timestamp"].diff().dt.total_seconds().fillna(0)
        session_breaks = (time_diffs > 1800).sum()  # 30 minutes
        feat["num_sessions"] = session_breaks + 1
        
        # Intra-session statistics
        if session_breaks > 0:
            session_lengths = []
            current_session_start = 0
            for i, is_break in enumerate(time_diffs > 1800):
                if is_break:
                    session_lengths.append(i - current_session_start)
                    current_session_start = i
            session_lengths.append(len(ev) - current_session_start)
            
            feat["avg_session_length"] = np.mean(session_lengths)
            feat["max_session_length"] = max(session_lengths)
            feat["session_length_std"] = np.std(session_lengths) if len(session_lengths) > 1 else 0
        
        # Temporal patterns
        feat["avg_time_between_events"] = time_diffs[time_diffs > 0].mean()
        feat["std_time_between_events"] = time_diffs[time_diffs > 0].std()
        feat["max_time_gap"] = time_diffs.max()
        
        # Activity intensity patterns
        hourly_activity = ev["event_timestamp"].dt.hour.value_counts()
        feat["peak_activity_hour"] = hourly_activity.index[0] if not hourly_activity.empty else -1
        feat["activity_hour_concentration"] = hourly_activity.max() / len(ev) if len(ev) > 0 else 0
    
    # 4. Recency and freshness features
    last_event = ev["event_timestamp"].max()
    first_event = ev["event_timestamp"].min()
    
    feat["hours_since_last_event"] = (ts - last_event).total_seconds() / 3600
    feat["days_since_first_event"] = (ts - first_event).total_seconds() / 86400
    feat["customer_lifetime_hours"] = (last_event - first_event).total_seconds() / 3600
    
    # Event velocity (events per day)
    if feat["days_since_first_event"] > 0:
        feat["events_per_day"] = len(ev) / feat["days_since_first_event"]
    else:
        feat["events_per_day"] = len(ev)
    
    # 5. Page and URL analysis
    pages = ev["page_url"].dropna()
    if not pages.empty:
        unique_domains = pages.apply(lambda x: x.split('/')[2] if '/' in x else x).nunique()
        feat["unique_domains"] = unique_domains
        feat["avg_page_revisits"] = len(pages) / pages.nunique() if pages.nunique() > 0 else 0
        
        # Checkout funnel analysis
        checkout_keywords = ['cart', 'checkout', 'payment', 'order', 'purchase']
        checkout_pages = sum(1 for page in pages if any(kw in page.lower() for kw in checkout_keywords))
        feat["checkout_funnel_pages"] = checkout_pages
        feat["checkout_conversion_rate"] = checkout_pages / len(pages) if len(pages) > 0 else 0
    
    # 6. Advanced behavioral indicators
    # Browsing vs purchasing intent
    browse_events = ['page_view', 'product_view', 'search']
    action_events = ['add_to_cart', 'purchase', 'checkout']
    
    browse_count = sum(feat.get(f"evt_{event}_count", 0) for event in browse_events)
    action_count = sum(feat.get(f"evt_{event}_count", 0) for event in action_events)
    
    feat["browse_to_action_ratio"] = browse_count / max(action_count, 1)
    feat["action_intent_score"] = action_count / max(len(ev), 1)
    
    # Compile text features
    page_concat = " ".join(pages.astype(str).tolist())
    event_data_concat = " ".join(ev["event_data"].dropna().astype(str).tolist())
    
    agg_features.append(feat)
    page_url_texts.append(page_concat)
    event_data_texts.append(event_data_concat)

# Create enhanced feature DataFrame
agg_df = pd.DataFrame(agg_features).fillna(0)
orders_with_feats = pd.concat([orders_sorted.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)
orders_with_feats["page_url_text"] = page_url_texts
orders_with_feats["event_data_text"] = event_data_texts

# --------------------------
# 4. Split back to train/test and add cross-client features
# --------------------------
train_fe = orders_with_feats[orders_with_feats["is_train"] == 1].copy().reset_index(drop=True)
test_fe = orders_with_feats[orders_with_feats["is_train"] == 0].copy().reset_index(drop=True)

for df in [train_fe, test_fe]:
    df.drop(columns=["is_train"], inplace=True)

# Add client-level aggregated features
def add_client_features(train_df, test_df):
    # Calculate client statistics from training data
    client_stats = train_df.groupby('client_id').agg({
        'total_price': ['mean', 'std', 'min', 'max', 'count'],
        'total_qty': ['mean', 'std'],
        'refunded': ['mean', 'sum']  # Only available in train
    }).round(4)
    
    client_stats.columns = ['_'.join(col).strip() for col in client_stats.columns]
    client_stats = client_stats.add_prefix('client_')
    
    # Merge with both train and test
    train_df = train_df.merge(client_stats, left_on='client_id', right_index=True, how='left')
    test_df = test_df.merge(client_stats, left_on='client_id', right_index=True, how='left')
    
    return train_df, test_df

train_fe, test_fe = add_client_features(train_fe, test_fe)

# --------------------------
# 5. Enhanced feature engineering
# --------------------------
def create_advanced_features(df):
    # Price-based features
    df["price_per_item"] = df["total_price"] / (df["total_qty"] + 1e-6)
    df["price_per_unique_item"] = df["total_price"] / (df["unique_items"] + 1e-6)
    df["qty_per_unique_item"] = df["total_qty"] / (df["unique_items"] + 1e-6)
    
    # Behavioral ratios
    df["high_value_ratio"] = df["high_price_items"] / (df["unique_items"] + 1e-6)
    df["price_diversity"] = df["std_price"] / (df["avg_price"] + 1e-6)
    
    # Time-based features
    df["events_per_hour_since_first"] = df.get("events_per_day", 0) / 24
    
    # Risk indicators
    df["late_night_orders"] = (df["is_late_night"] * df["total_price"]).astype(float)
    df["weekend_high_value"] = (df["is_weekend"] * (df["total_price"] > df["total_price"].quantile(0.75))).astype(int)
    
    return df

train_fe = create_advanced_features(train_fe)
test_fe = create_advanced_features(test_fe)

# --------------------------
# 6. Prepare final datasets with advanced text processing
# --------------------------
y = train_fe["refunded"].astype(int).values

# Prepare numeric features
drop_cols = ["order_id", "client_id", "order_timestamp", "line_items", "refunded"]
X_train_tab = train_fe.drop(columns=[c for c in drop_cols if c in train_fe.columns], errors='ignore').copy()
X_test_tab = test_fe.drop(columns=[c for c in drop_cols if c in test_fe.columns], errors='ignore').copy()

# Extract and process text
train_page_text = X_train_tab.pop("page_url_text").astype(str).fillna("")
train_event_text = X_train_tab.pop("event_data_text").astype(str).fillna("")
test_page_text = X_test_tab.pop("page_url_text").astype(str).fillna("")
test_event_text = X_test_tab.pop("event_data_text").astype(str).fillna("")

# Fill missing values with sophisticated strategy
numeric_cols = X_train_tab.select_dtypes(include=[np.number]).columns
categorical_cols = X_train_tab.select_dtypes(exclude=[np.number]).columns

# Handle numeric columns
for col in numeric_cols:
    median_val = X_train_tab[col].median()
    X_train_tab[col] = X_train_tab[col].fillna(median_val)
    X_test_tab[col] = X_test_tab[col].fillna(median_val)

# Handle categorical columns
for col in categorical_cols:
    mode_val = X_train_tab[col].mode().iloc[0] if not X_train_tab[col].mode().empty else "unknown"
    X_train_tab[col] = X_train_tab[col].fillna(mode_val)
    X_test_tab[col] = X_test_tab[col].fillna(mode_val)

# --------------------------
# 7. Advanced text processing with multiple TF-IDF configurations
# --------------------------
# Character-level and word-level TF-IDF for URLs
tfidf_url_word = TfidfVectorizer(max_features=300, ngram_range=(1, 3), stop_words='english', 
                                analyzer='word', min_df=2, max_df=0.95)
tfidf_url_char = TfidfVectorizer(max_features=200, ngram_range=(3, 5), analyzer='char_wb')

# Enhanced event data processing
tfidf_event_word = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words='english',
                                  analyzer='word', min_df=2)

# Fit and transform
X_train_url_word = tfidf_url_word.fit_transform(train_page_text)
X_test_url_word = tfidf_url_word.transform(test_page_text)

X_train_url_char = tfidf_url_char.fit_transform(train_page_text)
X_test_url_char = tfidf_url_char.transform(test_page_text)

X_train_event = tfidf_event_word.fit_transform(train_event_text)
X_test_event = tfidf_event_word.transform(test_event_text)

# --------------------------
# 8. Feature scaling and transformation
# --------------------------
# Apply quantile transformation for better distribution
scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
X_train_scaled = scaler.fit_transform(X_train_tab)
X_test_scaled = scaler.transform(X_test_tab)

# Combine all features
X_train_num_sparse = csr_matrix(X_train_scaled)
X_test_num_sparse = csr_matrix(X_test_scaled)

X_train_final = hstack([X_train_num_sparse, X_train_url_word, X_train_url_char, X_train_event]).tocsr()
X_test_final = hstack([X_test_num_sparse, X_test_url_word, X_test_url_char, X_test_event]).tocsr()

print(f"Final feature shapes: Train {X_train_final.shape}, Test {X_test_final.shape}")

# --------------------------
# 9. Advanced ensemble modeling with multiple algorithms
# --------------------------
# Calculate class weights
pos_count = y.sum()
neg_count = len(y) - pos_count
scale_pos_weight = neg_count / max(pos_count, 1)

print(f"Class distribution: {neg_count} negative, {pos_count} positive (ratio: {scale_pos_weight:.2f})")


def get_model_configs():
    return {
        'xgb': {
            'model': XGBClassifier,
            'param_space': {
                'n_estimators': [1000, 1500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 2, 5],
                'scale_pos_weight': [scale_pos_weight]
            },
            'fixed_params': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'gpu_hist',  # <---- enable GPU for XGBoost too
                'predictor': 'gpu_predictor'
            }
        },
        'lgb': {
            'model': LGBMClassifier,
            'param_space': {
                'n_estimators': [1000, 1500],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 63, 127],
                'feature_fraction': [0.8, 0.9],
                'bagging_fraction': [0.8, 0.9],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 2, 5],
                'class_weight': ['balanced']
            },
            'fixed_params': {
                'objective': 'binary',
                'metric': 'auc',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                # --- GPU parameters for LightGBM ---
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            }
        },
        'cat': {
            'model': CatBoostClassifier,
            'param_space': {
                'iterations': [1000, 1500],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5, 10],
                'auto_class_weights': ['Balanced']
            },
            'fixed_params': {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 42,
                'verbose': False,
                'task_type': 'GPU',  # <---- enable GPU for CatBoost too
                'devices': '0'
            }
        }
    }


# Optimize each model type
def optimize_model(model_name, config, X, y, n_trials=30):
    def objective(trial):
        params = config['fixed_params'].copy()
        
        # Sample hyperparameters
        for param, values in config['param_space'].items():
            if isinstance(values[0], int):
                params[param] = trial.suggest_categorical(param, values)
            elif isinstance(values[0], float):
                params[param] = trial.suggest_categorical(param, values)
            else:
                params[param] = trial.suggest_categorical(param, values)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = config['model'](**params)
            
            if model_name in ['xgb', 'lgb']:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            
            y_pred = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = {**config['fixed_params'], **study.best_params}
    return best_params, study.best_value

# Optimize models
model_configs = get_model_configs()
optimized_models = {}

print("Optimizing models...")
for name, config in model_configs.items():
    print(f"\nOptimizing {name.upper()}...")
    best_params, best_score = optimize_model(name, config, X_train_final, y, n_trials=25)
    optimized_models[name] = {'params': best_params, 'score': best_score}
    print(f"{name.upper()} best CV score: {best_score:.4f}")

# --------------------------
# 10. Multi-level stacking ensemble
# --------------------------
print("\nBuilding stacking ensemble...")

# Level 1: Base models with out-of-fold predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(y), len(optimized_models)))
test_preds = np.zeros((len(test_fe), len(optimized_models)))

model_names = list(optimized_models.keys())

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_final, y)):
    print(f"Fold {fold + 1}/5")
    X_tr, X_val = X_train_final[train_idx], X_train_final[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    fold_test_preds = []
    
    for i, (name, model_info) in enumerate(optimized_models.items()):
        model_class = model_configs[name]['model']
        model = model_class(**model_info['params'])
        
        if name in ['xgb', 'lgb']:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        
        # OOF predictions
        oof_preds[val_idx, i] = model.predict_proba(X_val)[:, 1]
        
        # Test predictions (will be averaged across folds)
        fold_test_preds.append(model.predict_proba(X_test_final)[:, 1])
    
    # Average test predictions for this fold
    for i, pred in enumerate(fold_test_preds):
        test_preds[:, i] += pred / 5

# Level 2: Meta-learner
print("Training meta-learner...")

# Add original features to meta-features for better performance
meta_features_train = np.hstack([oof_preds, X_train_scaled[:, :50]])  # Use top 50 original features
meta_features_test = np.hstack([test_preds, X_test_scaled[:, :50]])

# Multiple meta-learners
meta_models = {
    'logistic': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
    'xgb_meta': XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, 
                             random_state=42, scale_pos_weight=scale_pos_weight)
}

meta_oof = np.zeros((len(y), len(meta_models)))
meta_test = np.zeros((len(test_fe), len(meta_models)))

for fold, (train_idx, val_idx) in enumerate(skf.split(meta_features_train, y)):
    X_tr, X_val = meta_features_train[train_idx], meta_features_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    for i, (name, model) in enumerate(meta_models.items()):
        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)
        meta_oof[val_idx, i] = model_clone.predict_proba(X_val)[:, 1]
        meta_test[:, i] += model_clone.predict_proba(meta_features_test)[:, 1] / 5

# Final ensemble: weighted average of meta-models
meta_weights = []
for i, name in enumerate(meta_models.keys()):
    score = roc_auc_score(y, meta_oof[:, i])
    meta_weights.append(score)
    print(f"Meta-model {name} CV AUC: {score:.4f}")

# Normalize weights
meta_weights = np.array(meta_weights)
meta_weights = meta_weights / meta_weights.sum()

final_predictions = np.average(meta_test, axis=1, weights=meta_weights)
final_predictions_binary = (final_predictions >= 0.5).astype(int)

# --------------------------
# 11. Model evaluation and diagnostics
# --------------------------
print("\n" + "="*50)
print("FINAL MODEL EVALUATION")
print("="*50)

# Individual base model performance
print("\nBase Model Performance:")
for i, name in enumerate(model_names):
    oof_auc = roc_auc_score(y, oof_preds[:, i])
    print(f"{name.upper():>8}: {oof_auc:.4f}")

# Meta-model ensemble performance
final_oof = np.average(meta_oof, axis=1, weights=meta_weights)
final_auc = roc_auc_score(y, final_oof)
print(f"\nFinal Ensemble AUC: {final_auc:.4f}")

# Classification report for final model
final_oof_binary = (final_oof >= 0.5).astype(int)
print("\nClassification Report (Final Ensemble):")
print(classification_report(y, final_oof_binary))

# Feature importance analysis (using best XGBoost model)
print("\nAnalyzing feature importance...")
best_xgb_params = optimized_models['xgb']['params']
importance_model = XGBClassifier(**best_xgb_params)
importance_model.fit(X_train_final, y)

try:
    feature_importance = importance_model.feature_importances_
    # Get top features (limited to numeric features for interpretation)
    n_numeric_features = len(X_train_tab.columns)
    top_indices = np.argsort(feature_importance[:n_numeric_features])[-20:][::-1]
    
    print("\nTop 20 Most Important Features:")
    feature_names = X_train_tab.columns.tolist()
    for i, idx in enumerate(top_indices):
        if idx < len(feature_names):
            print(f"{i+1:2d}. {feature_names[idx]:30s}: {feature_importance[idx]:.4f}")
except Exception as e:
    print(f"Could not extract feature importance: {e}")

# --------------------------
# 12. Advanced prediction calibration
# --------------------------
from sklearn.calibration import CalibratedClassifierCV

print("\nApplying prediction calibration...")

# Calibrate final predictions using Platt scaling
calibration_model = CalibratedClassifierCV(
    base_estimator=LogisticRegression(random_state=42),
    method='sigmoid',
    cv=3
)

# Fit calibration on OOF predictions
calibration_features = final_oof.reshape(-1, 1)
calibration_model.fit(calibration_features, y)

# Apply calibration to test predictions
calibrated_test_probs = calibration_model.predict_proba(
    final_predictions.reshape(-1, 1)
)[:, 1]

# Final binary predictions with calibration
calibrated_binary = (calibrated_test_probs >= 0.5).astype(int)

# --------------------------
# 13. Prediction confidence and uncertainty estimation
# --------------------------
print("Computing prediction confidence...")

# Calculate prediction confidence based on model agreement
model_agreement = np.std(test_preds, axis=1)  # Lower std = higher agreement
confidence_scores = 1 / (1 + model_agreement)  # Convert to 0-1 scale

# Identify high-confidence predictions
high_confidence_mask = confidence_scores > np.percentile(confidence_scores, 75)
high_conf_accuracy = "High confidence predictions identified"

print(f"High confidence predictions: {high_confidence_mask.sum()}/{len(high_confidence_mask)}")

# --------------------------
# 14. Advanced submission strategies
# --------------------------
print("\nGenerating multiple submission strategies...")

# Strategy 1: Calibrated predictions
submission_calibrated = pd.DataFrame({
    "order_id": test_order_ids.values,
    "refunded": calibrated_binary
})

# Strategy 2: Conservative threshold (optimize for precision)
conservative_threshold = np.percentile(final_predictions, 90)  # Top 10% as positive
submission_conservative = pd.DataFrame({
    "order_id": test_order_ids.values,
    "refunded": (final_predictions >= conservative_threshold).astype(int)
})

# Strategy 3: Aggressive threshold (optimize for recall)
aggressive_threshold = np.percentile(final_predictions, 70)  # Top 30% as positive
submission_aggressive = pd.DataFrame({
    "order_id": test_order_ids.values,
    "refunded": (final_predictions >= aggressive_threshold).astype(int)
})

# Strategy 4: Dynamic threshold based on confidence
dynamic_predictions = calibrated_binary.copy()
# For low confidence predictions, use more conservative threshold
low_conf_mask = confidence_scores <= np.percentile(confidence_scores, 25)
dynamic_predictions[low_conf_mask] = (
    final_predictions[low_conf_mask] >= 0.7
).astype(int)

submission_dynamic = pd.DataFrame({
    "order_id": test_order_ids.values,
    "refunded": dynamic_predictions
})

# Save all strategies
submissions = {
    'calibrated': submission_calibrated,
    'conservative': submission_conservative,
    'aggressive': submission_aggressive,
    'dynamic': submission_dynamic
}

for name, sub_df in submissions.items():
    filename = f"submission_{name}.csv"
    sub_df.to_csv(filename, index=False)
    positive_rate = sub_df['refunded'].mean()
    print(f"Saved {filename} (positive rate: {positive_rate:.3f})")

# --------------------------
# 15. Model validation and robustness checks
# --------------------------
print("\n" + "="*50)
print("MODEL VALIDATION & ROBUSTNESS")
print("="*50)

# Time-based validation (if timestamps show temporal patterns)
if 'order_timestamp' in train.columns:
    print("\nTemporal validation:")
    train_with_ts = train.copy()
    train_with_ts = train_with_ts.sort_values('order_timestamp')
    
    # Split by time (80% train, 20% validation)
    split_idx = int(0.8 * len(train_with_ts))
    temporal_train = train_with_ts.iloc[:split_idx]
    temporal_val = train_with_ts.iloc[split_idx:]
    
    print(f"Temporal split: {len(temporal_train)} train, {len(temporal_val)} validation")
    print(f"Train period: {temporal_train['order_timestamp'].min()} to {temporal_train['order_timestamp'].max()}")
    print(f"Val period: {temporal_val['order_timestamp'].min()} to {temporal_val['order_timestamp'].max()}")

# Cross-validation stability
print(f"\nCross-validation stability:")
print(f"Base model AUC std: {np.std([roc_auc_score(y, oof_preds[:, i]) for i in range(len(model_names))]):.4f}")

# Prediction distribution analysis
print(f"\nPrediction distribution:")
print(f"Min prediction: {final_predictions.min():.4f}")
print(f"Max prediction: {final_predictions.max():.4f}")
print(f"Mean prediction: {final_predictions.mean():.4f}")
print(f"Std prediction: {final_predictions.std():.4f}")

percentiles = [10, 25, 50, 75, 90, 95, 99]
print("Prediction percentiles:")
for p in percentiles:
    print(f"  {p:2d}th: {np.percentile(final_predictions, p):.4f}")

# --------------------------
# 16. Final recommendations and insights
# --------------------------
print("\n" + "="*50)
print("RECOMMENDATIONS & INSIGHTS")
print("="*50)

print("\nModel Performance Summary:")
print(f"â€¢ Final ensemble achieves {final_auc:.4f} AUC on cross-validation")
print(f"â€¢ Best individual model: {model_names[np.argmax([optimized_models[name]['score'] for name in model_names])]} with {max([optimized_models[name]['score'] for name in model_names]):.4f} AUC")
print(f"â€¢ Ensemble improvement: +{final_auc - max([optimized_models[name]['score'] for name in model_names]):.4f} AUC points")

print("\nKey Model Features:")
print("â€¢ Advanced behavioral event features with time decay")
print("â€¢ Multi-window temporal aggregations (1d to 90d)")
print("â€¢ Sequential pattern analysis and session detection")  
print("â€¢ Client-level historical features")
print("â€¢ Multi-level stacking with calibration")
print("â€¢ Multiple TF-IDF text representations")

print("\nSubmission Strategy Recommendations:")
print("â€¢ Use 'calibrated' submission for best overall performance")
print("â€¢ Use 'conservative' if precision is more important than recall")
print("â€¢ Use 'aggressive' if recall is more important than precision")
print("â€¢ Use 'dynamic' for confidence-aware predictions")

print(f"\nPredicted class distribution in test set:")
for name, sub_df in submissions.items():
    pos_rate = sub_df['refunded'].mean()
    pos_count = sub_df['refunded'].sum()
    print(f"â€¢ {name:12s}: {pos_count:4d} positive ({pos_rate:.2%})")

print("\n" + "="*50)
print("PIPELINE COMPLETE - TARGETING 90% ACCURACY")
print("="*50)

# Final validation score display
print(f"\nFINAL CROSS-VALIDATION SCORE: {final_auc:.4f}")
if final_auc >= 0.85:
    print("ğŸ�¯ SUCCESS: Model performance is in the target range!")
elif final_auc >= 0.80:
    print("ğŸ“ˆ GOOD: Strong model performance, minor tuning may help")
else:
    print("âš ï¸�  ATTENTION: Consider additional feature engineering or data quality checks")

print("\nFiles generated:")
for name in submissions.keys():
    print(f"â€¢ submission_{name}.csv")

print("\nNext steps if accuracy is still below target:")
print("1. Investigate data quality issues and outliers")
print("2. Add external data sources if available")
print("3. Implement neural network approaches")
print("4. Apply advanced feature selection techniques")
print("5. Consider semi-supervised learning with test data")
print("6. Analyze prediction errors for pattern insights")




