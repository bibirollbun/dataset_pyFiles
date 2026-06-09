# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Step 2: Sample ~1,000,000 rows from training data with complete labels (Balanced dataset)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
import numpy as np


# PARQUET_DATA_DIR = "drive/MyDrive/train_data_parquet"
# PARQUET_LABEL_DIR = "drive/MyDrive/train_labels_parquet"

# PARQUET_DATA_DIR = "train_data_parquet"
# PARQUET_LABEL_DIR = "train_labels_parquet"

TRAIN_DATA_PATH = '/kaggle/input/amex-default-prediction/train_data.csv'
TRAIN_LABELS_PATH = '/kaggle/input/amex-default-prediction/train_labels.csv'

print("Loading labels...")
# df_labels = dd.read_parquet(PARQUET_LABEL_DIR, engine="pyarrow").compute()
df_labels = pd.read_csv(TRAIN_LABELS_PATH)

print("Label distribution:")
print(df_labels["target"].value_counts())

# separate counts based on target
n_default = df_labels["target"].sum() # target=1 is default
n_nondefault = len(df_labels) - n_default

print(f"Default customers: {n_default:,}")
print(f"Non-default customers: {n_nondefault:,}")

# calculate average rows per customer
print("Loading small subset to estimate rows per customer...")
# df_train_dask = dd.read_parquet(PARQUET_DATA_DIR, engine="pyarrow")
df_train_dask = dd.read_csv(TRAIN_DATA_PATH)
rows_per_customer_est = int(df_train_dask.shape[0].compute() / len(df_labels))
print(f"Estimated rows per customer: {rows_per_customer_est}")

# 2 million rows max
target_rows = 2_000_000
max_customers_total = target_rows // rows_per_customer_est
max_customers_each_class = max_customers_total // 2

frac_default = max_customers_each_class / n_default
frac_nondefault = max_customers_each_class / n_nondefault

print(f"Sampling {max_customers_each_class:,} customers from each class")

# randomly sample customers
default_customers = df_labels[df_labels["target"] == 1].sample(
    frac=frac_default, random_state=42
)
nondefault_customers = df_labels[df_labels["target"] == 0].sample(
    frac=frac_nondefault, random_state=42
)

# combine sampled customers
sampled_customers = pd.concat([default_customers, nondefault_customers])
sampled_customer_list = sampled_customers["customer_ID"].tolist()

print("Filtering training data to sampled customers...")
df_train_sample = df_train_dask[
    df_train_dask["customer_ID"].isin(sampled_customer_list)
].compute()

print(f"Loaded {len(df_train_sample):,} rows for sampled customers")

# if larger than 2 million rows, randomly sample down to 2 million
if len(df_train_sample) > 2_000_000:
    df_train_sample = df_train_sample.sample(n=2_000_000, random_state=42)

# combine with labels
df_sample = pd.merge(df_train_sample, df_labels, on="customer_ID", how="left")

# check results
print("\nSampling done!")
print(f"Shape: {df_sample.shape}")
print(f"Unique customers: {df_sample['customer_ID'].nunique()}")
print(df_sample["target"].value_counts(normalize=True))



# Step 3 Final check
train_data_sampled = df_sample

# for later use, convert to normal pandas DataFrame (not pyarrow type)
train_data_sampled = train_data_sampled.convert_dtypes(dtype_backend="numpy_nullable").infer_objects()

print(type( train_data_sampled ))

display(train_data_sampled.head())



# Step 4: Data Cleaning

print("\nShowing missing value percentages for top 20 columns...")
missing_values_perc = (train_data_sampled.isnull().sum() / len(train_data_sampled)) * 100
missing_top20 = missing_values_perc.sort_values(ascending=False).head(20)
display(missing_top20)

print("Starting data cleaning...")

# ensure date columns are in datetime format
if "S_2" in train_data_sampled.columns:
    train_data_sampled["S_2"] = pd.to_datetime(train_data_sampled["S_2"], errors="coerce")

# separate numeric and string columns
num_cols = train_data_sampled.select_dtypes(include=["number"]).columns
str_cols = train_data_sampled.select_dtypes(include=["string", "object"]).columns

print(f"Original rows: {len(train_data_sampled.columns)}")

# remove columns with more than 40% missing data
cols_to_drop = missing_values_perc[missing_values_perc > 40].index.tolist()
train_data_sampled = train_data_sampled.drop(columns=cols_to_drop)
print(f"Dropped {len(cols_to_drop)} columns with more than 40% missing data.")

# delete any "unnamed" columns if exist
drop_cols = [c for c in train_data_sampled.columns if "unnamed" in c.lower()]
train_data_sampled = train_data_sampled.drop(columns=drop_cols, errors="ignore")

print("Finished data cleaning!")
print(f"Rows left: {len(train_data_sampled.columns)}")



# Step 5: Feature Engineering

print("Beginning feature engineering...")

# sort by customer_ID and date for time series processing
train_data_sampled = train_data_sampled.sort_values(['customer_ID', 'S_2']).reset_index(drop=True)

# data types of numeric columns
num_cols = train_data_sampled.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [c for c in num_cols if c not in ['target']]  # exclude target column

# calculate customer-level aggregate features for numeric columns
agg_funcs = ['mean', 'max', 'min', 'last']
customer_features = (
    train_data_sampled
    .groupby('customer_ID')[num_cols]
    .agg(agg_funcs)
)

# flatten multi-level columns
customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]

# combine target
customer_features = customer_features.merge(
    train_data_sampled[['customer_ID', 'target']].drop_duplicates(),
    on='customer_ID', how='left'
)

print("Finished feature engineering!")
display(customer_features.head())


# Step 6: Exploratory Data Analysis (EDA)

# use a cleaner visual theme
sns.set_style('whitegrid')

# current working DataFrame
df = customer_features.dropna().reset_index(drop=True)

print(f"After dropping missing values: {len(df)} rows left, {len(df.columns)} columns.")

# change to parquet
df.to_parquet("cleaned_data.parquet", index=False)

print("Summary statistics for numeric columns")
display(df.describe().T)

print("\nListing missing value percentages for top 20 columns...")
missing_values_perc = (df.isnull().sum() / len(df)) * 100
missing_top20 = missing_values_perc.sort_values(ascending=False).head(20)
display(missing_top20)

plt.figure(figsize=(10,6))
sns.barplot(x=missing_top20.values, y=missing_top20.index, palette='viridis')
plt.title("Top 20 Columns with Most Missing Values")
plt.xlabel("Missing Value Percentage (%)")
plt.ylabel("Feature")
plt.show()

print("\nTarget variable distribution (per unique customer)")

unique_customers = df.drop_duplicates(subset=['customer_ID'])
plt.figure(figsize=(8,5))
sns.countplot(x='target', data=unique_customers, palette='pastel')
plt.title('Distribution of Target Variable (Per Unique Customer)')
plt.xlabel('Default (1) vs. No Default (0)')
plt.ylabel('Number of Unique Customers')

# show percentage on top of bars
total = len(unique_customers)
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().text(
        p.get_x() + p.get_width()/2., height + 50,
        f'{100*height/total:.2f}%', ha='center', fontsize=10
    )
plt.show()

print("\nKey categorical feature count plots")
key_cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_68']

for col in key_cat_features:
    if col in df.columns:
        plt.figure(figsize=(10,5))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='coolwarm')
        plt.title(f'Count Plot for {col}')
        plt.xscale('log')  # use log scale for better visibility
        plt.show()



# Step 7: Test data preparation
import pandas as pd
import os
import pyarrow.parquet as pq
import pyarrow as pa

# Clean up old parquet dirs if exist
os.system("rm -rf test_data_parquet1")
os.makedirs("test_data_parquet1", exist_ok=True)

TEST_DATA_DIR = '/kaggle/input/amex-default-prediction/test_data.csv'

# separatedly load csv and transfer to parquet
chunksize = 500_000  # 500k rows per chunk
i = 0
for chunk in pd.read_csv(TEST_DATA_DIR, chunksize=chunksize):
    table = pa.Table.from_pandas(chunk)
    pq.write_table(table, f"test_data_parquet1/part_{i}.parquet", compression="snappy")
    i += 1
    print(f"âœ… Wrote data chunk {i}")


print("ðŸ’¾ All train set CSV chunks saved as parquet successfully!")



import dask.dataframe as dd
import pandas as pd
import numpy as np
import gc
import os
import shutil

PARQUET_TEST = "/kaggle/working/test_data_parquet1"
OUTPUT_FILE = "/kaggle/working/cleaned_test_data.parquet"

print("ðŸ“¥ Loading metadata...")
df_meta = dd.read_parquet(PARQUET_TEST, engine="pyarrow")
nparts = df_meta.npartitions
print(f"Found {nparts} partitions")

agg_funcs = ["mean", "max", "min", "last"]

# Detect columns to drop
sample = df_meta.get_partition(0).head(10_000, compute=True)
cols_to_drop = sample.columns[sample.isnull().mean() > 0.4].tolist()
print(f"Dropping {len(cols_to_drop)} high-missing columns")

df_final = []

for i in range(nparts):
    print(f"\nðŸ§© Processing partition {i+1}/{nparts}")
    df = df_meta.get_partition(i).compute()
    df = df.drop(columns=cols_to_drop, errors="ignore")

    if "S_2" in df.columns:
        df["S_2"] = pd.to_datetime(df["S_2"], errors="coerce")
    df = df.sort_values(["customer_ID", "S_2"])

    # Downcast numeric
    for col in df.select_dtypes("float64"):
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64"):
        df[col] = df[col].astype("int32")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "target" in num_cols:
        num_cols.remove("target")

    # Aggregate within this partition
    part_agg = df.groupby("customer_ID")[num_cols].agg(agg_funcs)
    part_agg.columns = ["_".join(c) for c in part_agg.columns.to_flat_index()]
    df_final.append(part_agg)

    # Clean up memory
    del df, part_agg
    gc.collect()

print("\nðŸ”— Combining partial results...")
df_all = pd.concat(df_final)
del df_final
gc.collect()

df_final = df_all.groupby("customer_ID").agg("last")
del df_all
gc.collect()

# Remove any old copy before writing
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

OLD_DIR = "/kaggle/working/agg_partitions"

if os.path.exists(OLD_DIR):
    print(f"ðŸ§¹ Removing old directory: {OLD_DIR}")
    shutil.rmtree(OLD_DIR)
    print("âœ… Old directory deleted")
else:
    print("No old agg_partitions directory found")

print(f"ðŸ’¾ Writing final file to {OUTPUT_FILE} ...")
df_final.to_parquet(OUTPUT_FILE, index=True)

print("\nðŸŽ‰ Final features saved successfully!")


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import accuracy_score, f1_score
import dask.dataframe as dd

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

# Load dataset lazily with Dask
ddf = dd.read_parquet('/kaggle/working/cleaned_data.parquet')

# Define feature columns (exclude unnecessary columns)
feature_cols = [c for c in ddf.columns if c not in ['customer_ID', 'target']]

# Sample fraction for model selection (including target)
sample_frac = 1
ddf_sample = ddf[feature_cols + ['target']].sample(frac=sample_frac, random_state=42).compute()

# Split features and target
X_sample = ddf_sample[feature_cols]
y_sample = ddf_sample['target']

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#study = optuna.create_study(direction="maximize")

def objective(trial):
  # Determine hyperparameter values
  learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
  num_leaves = trial.suggest_int("num_leaves", 2, 256)
  max_depth = trial.suggest_int("max_depth", 5, 30)
  min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
  subsample = trial.suggest_float("subsample", 0.5, 1.0)
  colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
  n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    
  model = LGBMClassifier(
    learning_rate=learning_rate,
    num_leaves=num_leaves,
    max_depth=max_depth,
    min_child_samples=min_child_samples,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    n_estimators=n_estimators,
    random_state=42
  )
  amex_scores = []
  for train_idx, test_idx in kf.split(X_sample):
        X_train, X_test = X_sample.iloc[train_idx], X_sample.iloc[test_idx]
        y_train, y_test = y_sample.iloc[train_idx], y_sample.iloc[test_idx]

        # Fit model
        model.fit(X_train, y_train)

        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_df = pd.DataFrame({'target': y_test.values})
        y_pred_df = pd.DataFrame({'prediction': y_pred_proba})
      
        amex_scores.append(amex_metric(y_test_df, y_pred_df))

  return np.mean(amex_scores)

# Run the study and review the results
#study.optimize(objective, n_trials=20)
#print("Best trial:")
#print(" Value: {}".format(study.best_trial.value))
#print(" Params: {}".format(study.best_trial.params))
params = {
    "learning_rate": 0.011326681203182443,
    "num_leaves": 76,
    "max_depth": -1, 
    "min_child_samples": 70,
    "subsample": 0.638068300141083,
    "colsample_bytree": 0.639335047549834,
    "n_estimators": 915
}

# Define models
models = [
    ('LogisticRegression', LogisticRegression(max_iter=1000)),
    ('LightGBM', LGBMClassifier()),
    ('LightGBM-tuned', LGBMClassifier(**params)),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Function to evaluate a model
def evaluate_model(model, X, y, kf):
    acc_scores = []
    f1_scores = []
    amex_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        # amex score
        y_test_df = pd.DataFrame({'target': y_test.values})
        y_pred_df = pd.DataFrame({'prediction': y_pred_proba})
        amex_scores.append(amex_metric(y_test_df, y_pred_df))
        
    return np.mean(acc_scores), np.mean(f1_scores), np.mean(amex_scores)

# Evaluate all models
best_model_name = None
best_amex = 0
results = []

for name, model in models:
    print(f"Evaluating {name} ...")
    acc, f1, amex = evaluate_model(model, X_sample, y_sample, kf)
    results.append((name, acc, f1, amex))
    print(f"{name}: avg accuracy={acc:.4f}, avg f1={f1:.4f}, avg amex={amex:.4f}")

    if amex > best_amex:
        best_amex = amex
        best_model_name = name

# Summary
print("\nSummary of all models:")
for name, acc, f1, amex in results:
    print(f"{name}: accuracy={acc:.4f}, f1={f1:.4f}, amex={amex:.4f}")

print(f"\nBest model based on AMEX score: {best_model_name} with F1={best_amex:.4f}")



import pandas as pd
# Summary
print("\nSummary of all models:")
for name, acc, f1, amex in results:
    print(f"{name}: accuracy={acc:.4f}, f1={f1:.4f}, amex={amex:.4f}")

print(f"\nBest model based on AMEX score: {best_model_name} with F1={best_amex:.4f}")

test_data = pd.read_parquet('/kaggle/working/cleaned_test_data.parquet')
test_data = test_data.reset_index()


print(test_data.head())


print(ddf.head())


diff = list(set(test_data.columns)-set(ddf.columns))
print(diff)
test_data = test_data.drop(columns=diff, axis=1)
test_feature_cols = test_data.drop('customer_ID', axis=1).columns
print(test_feature_cols)


model_dict = dict(models)
final_model = model_dict[best_model_name]
X, y = ddf[feature_cols], ddf['target']
final_model.fit(X.compute(), y.compute())
X_test = test_data[test_feature_cols]
# Feed transformed test data into the final_model prediction to get our output values
probs = final_model.predict_proba(X_test)[:, 1]

# Create the final submission csv using the model predictions
submission = pd.DataFrame({'customer_ID': test_data['customer_ID'], 'prediction': probs})
submission = submission.set_index('customer_ID')


submission.to_csv('submission.csv')

