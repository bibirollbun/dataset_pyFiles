


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import os

import zipfile
import shutil

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


warnings.simplefilter(action="ignore", category=FutureWarning)


class CFG:
    input_path = Path('/kaggle/input')
    
    # Paths for ZIP files
    dataset_snuggle = 'flight-delays-fall-2018'
    train_zip = input_path / dataset_snuggle / "flight_delays_train.csv.zip"
    test_zip = input_path / dataset_snuggle / "flight_delays_test.csv.zip"
    submission_zip = input_path / dataset_snuggle / "sample_submission.csv.zip"

    working_dir = Path('/kaggle/working')
    # Ensure working directory exists
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # csv files
    train_file = working_dir / "flight_delays_train.csv"
    test_file = working_dir / "flight_delays_test.csv"
    submission_sample_file = working_dir / "sample_submission.csv"
    submission_file = working_dir / "submission.csv"

    # Definitions
    # debug = True # False
    debug = False # True
    
    cat_features = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
    num_features = ["DepTime", "Distance"]
    
    target = "dep_delayed_15min"
    
    sample_fraction = 0.05 if debug else 1
    random_state = 42
    train_size = 0.7
    test_valid_rel = 0.5
    
    scoring = "roc_auc"
    cv = 3  # 3-fold cross-validation
    verbose = 3
    
    thresholds = [0.5, 0.4, 0.3]
    threshold = 0.5
    best_params = {
        "n_estimators": 500,
        "max_depth": 50,
        "max_features": "sqrt",
        "min_samples_split": 10,
        "min_samples_leaf": 10,
        "class_weight": "balanced_subsample",
        "random_state": random_state,
        "n_jobs": -1
    }


# Configure Pandas to display each DataFrame row in one line without wrapping
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


for dirname, _, filenames in os.walk(CFG.input_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Extract ZIPs to working directory
for zip_file in [CFG.train_zip, CFG.test_zip, CFG.submission_zip]:
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(CFG.working_dir)

# Move extracted files to working directory only if they are inside a subfolder
for extracted_file in CFG.working_dir.glob(f'{CFG.dataset_snuggle}/*.csv'):
    dest_file = CFG.working_dir / extracted_file.name
    if not dest_file.exists():  # Move only if the file is not already in working_dir
        shutil.move(str(extracted_file), str(dest_file))


# Load datasets
train_df = pd.read_csv(CFG.train_file)
test_df = pd.read_csv(CFG.test_file)
submission_df = pd.read_csv(CFG.submission_sample_file)

# Display first rows and shapes
print('train_df')
print(f"Train shape: {train_df.shape}")
print(train_df.head())
print()
print('-'*45)
print()
print('test_df')
print(f"Test shape: {test_df.shape}")
print(test_df.head())
print()
print('-'*45)
print()
print('submission_df')
print(submission_df.head())


train_df.info()
print()
print('-'*45)
print()
test_df.info()


# Set pastel color palette
sns.set_palette("pastel")


# Convert DepTime to decimal hours
train_df["DepTime_hours"] = train_df["DepTime"] // 100 + (train_df["DepTime"] % 100) / 60
test_df["DepTime_hours"] = test_df["DepTime"] // 100 + (test_df["DepTime"] % 100) / 60

# Plot DepTime distribution
plt.figure(figsize=(8, 5))
sns.kdeplot(train_df["DepTime_hours"], label="Train", linewidth=2, fill=True, alpha=0.3)
sns.kdeplot(test_df["DepTime_hours"], label="Test", linewidth=2, fill=True, alpha=0.3)
plt.xticks(range(0, 25, 1))
plt.xlabel("Departure Time (Hours)")
plt.ylabel("Density")
plt.title("Distribution of Departure Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()



# KS Test for DepTime (after conversion)
ks_stat_dep, p_value_dep = ks_2samp(train_df["DepTime_hours"], test_df["DepTime_hours"])
print(f"KS test for Departure Time: statistic={ks_stat_dep:.4f}, p-value={p_value_dep:.4f}")


plt.figure(figsize=(8, 5))
sns.kdeplot(train_df["Distance"], label="Train", linewidth=2, fill=True, alpha=0.3)
sns.kdeplot(test_df["Distance"], label="Test", linewidth=2, fill=True, alpha=0.3)
plt.xlabel("Distance")
plt.ylabel("Density")
plt.title("Distribution of Distance")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# KS Test for Distance
ks_stat_dist, p_value_dist = ks_2samp(train_df["Distance"], test_df["Distance"])
print(f"KS test for Distance: statistic={ks_stat_dist:.4f}, p-value={p_value_dist:.4f}")


# Plot categorical feature distributions
cat_features = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index, ax=axes[i], alpha=0.6, label="Train")
    sns.countplot(data=test_df, x=col, order=test_df[col].value_counts().index, ax=axes[i], alpha=0.6, label="Test")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].tick_params(axis='x', rotation=90)
    axes[i].legend()

plt.tight_layout()
plt.show()


def compute_js_divergence(train_col, test_col):
    """Computes Jensen-Shannon divergence between train and test distributions"""
    train_dist = train_col.value_counts(normalize=True).sort_index()
    test_dist = test_col.value_counts(normalize=True).sort_index()
    
    # Align indices to avoid mismatches
    all_categories = train_dist.index.union(test_dist.index)
    train_dist = train_dist.reindex(all_categories, fill_value=0)
    test_dist = test_dist.reindex(all_categories, fill_value=0)
    
    return jensenshannon(train_dist, test_dist)


cat_features = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

# Compute JS divergence for each categorical feature
js_results = {col: compute_js_divergence(train_df[col], test_df[col]) for col in cat_features}

# Print results
for col, js_div in js_results.items():
    print(f"JS Divergence for {col}: {js_div:.4f}")


# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=train_df["dep_delayed_15min"], palette="pastel")
plt.title("Target Distribution (Train Data)")
plt.xticks([0, 1], ["Not Delayed (0)", "Delayed (1)"])
plt.show()


# Set pastel color palette
sns.set_palette("pastel")

# Numerical features comparison
num_features = ["DepTime_hours", "Distance"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, col in enumerate(num_features):
    sns.kdeplot(data=train_df, x=col, hue="dep_delayed_15min", fill=True, ax=axes[i], alpha=0.4)
    axes[i].set_title(f"Distribution of {col} by Delay Status")
    axes[i].legend(["Not Delayed (0)", "Delayed (1)"])

plt.tight_layout()
plt.show()


# Set pastel color palette
sns.set_palette("pastel")

# Numerical features comparison (Corrected)
num_features = ["DepTime_hours", "Distance"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, col in enumerate(num_features):
    sns.kdeplot(data=train_df, x=col, hue="dep_delayed_15min", fill=True, ax=axes[i], alpha=0.4, common_norm=False)
    axes[i].set_title(f"Distribution of {col} by Delay Status")
    axes[i].legend(["Not Delayed (0)", "Delayed (1)"])

plt.tight_layout()
plt.show()



# Create a temporary column for visualization
train_df["delay_temp"] = train_df["dep_delayed_15min"].map({"N": 0, "Y": 1})

# Categorical features comparison
cat_features = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    delay_rate = train_df.groupby(col)["delay_temp"].mean().sort_values()
    
    # Convert index to string to avoid issues with numeric indices
    delay_rate.index = delay_rate.index.astype(str)

    sns.barplot(x=delay_rate.index, y=delay_rate.values, ax=axes[i])
    axes[i].set_title(f"Average Delay Rate by {col}")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)

plt.tight_layout()
plt.show()

# Drop temporary column
train_df.drop(columns=["delay_temp"], inplace=True)



# Create a temporary column for visualization
train_df["delay_temp"] = train_df["dep_delayed_15min"].map({"N": 0, "Y": 1})

# Compute average delay rate for Origin and Dest
origin_delay = train_df.groupby("Origin")["delay_temp"].mean().sort_values()
dest_delay = train_df.groupby("Dest")["delay_temp"].mean().sort_values()

# Filter airports with extreme delay rates and sort them
low_delay_origin = origin_delay[origin_delay < 0.03].sort_values()
high_delay_origin = origin_delay[origin_delay > 0.6].sort_values()

low_delay_dest = dest_delay[dest_delay < 0.06].sort_values()
high_delay_dest = dest_delay[dest_delay > 0.6].sort_values()

# Plot the filtered results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.barplot(x=low_delay_origin.index, y=low_delay_origin.values, ax=axes[0, 0])
axes[0, 0].set_title("Origins with Low Delay Rate (<6%)")
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=90)

sns.barplot(x=high_delay_origin.index, y=high_delay_origin.values, ax=axes[0, 1])
axes[0, 1].set_title("Origins with High Delay Rate (>60%)")
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=90)

sns.barplot(x=low_delay_dest.index, y=low_delay_dest.values, ax=axes[1, 0])
axes[1, 0].set_title("Destinations with Low Delay Rate (<6%)")
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=90)

sns.barplot(x=high_delay_dest.index, y=high_delay_dest.values, ax=axes[1, 1])
axes[1, 1].set_title("Destinations with High Delay Rate (>60%)")
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=90)

plt.tight_layout()
plt.show()

# Drop temporary column
train_df.drop(columns=["delay_temp"], inplace=True)



print(train_df.columns)


# Sample dataset (stratified)
stratify_cols = train_df[[CFG.target, "UniqueCarrier"]].astype(str)
if CFG.debug:
    train_sample, _ = train_test_split(
        train_df, 
        test_size = (1 - CFG.sample_fraction), 
        stratify = stratify_cols, 
        random_state = CFG.random_state
    )
else:
    train_sample = train_df.copy()

# Target conversion
y_rf = train_sample[CFG.target].map({"N": 0, "Y": 1})


# Preprocessor (One-Hot Encoding for categorical features)
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
], remainder="passthrough")


# Apply transformation
X_transformed = preprocessor.fit_transform(train_sample.drop(columns=[CFG.target]))


# Extract feature names
final_feature_names = preprocessor.get_feature_names_out()


# Convert transformed array back to DataFrame
X_rf = pd.DataFrame(X_transformed, columns=final_feature_names)


X_train, X_temp, y_train, y_temp = train_test_split(X_rf, y_rf, train_size = CFG.train_size, stratify = y_rf, random_state = CFG.random_state)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size = CFG.test_valid_rel, stratify=y_temp, random_state = CFG.random_state)


rf_best = RandomForestClassifier(**CFG.best_params)
rf_best.fit(X_train, y_train)


y_pred_proba = rf_best.predict_proba(X_valid)[:, 1]


for cut in CFG.thresholds:
    print(f"\n### Threshold: {cut} ###")
    
    # Convert probabilities to binary predictions
    y_valid_pred = (y_pred_proba >= cut).astype(int)
    
    # Evaluation Metrics
    roc_auc = roc_auc_score(y_valid, y_pred_proba)
    conf_matrix = confusion_matrix(y_valid, y_valid_pred)

    print(f"ROC AUC: {roc_auc:.4f}")
    print("Classification Report:\n", classification_report(y_valid, y_valid_pred))

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Delay", "Delay"], yticklabels=["No Delay", "Delay"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Threshold = {cut})")
    plt.show()


# Apply the same preprocessing to the test dataset
X_test_transformed = preprocessor.transform(test_df)

# Convert transformed array back to a DataFrame
X_test_final = pd.DataFrame(X_test_transformed, columns=final_feature_names)

# Generate predictions
y_test_proba = rf_best.predict_proba(X_test_final)[:, 1]


# Create the submission DataFrame
submission_df = pd.DataFrame({
    "id": range(len(y_test_proba)),
    "dep_delayed_15min": y_test_proba
})

# Save as CSV
submission_csv_path = CFG.submission_file
submission_df.to_csv(submission_csv_path, index=False)


print(f"Submission file saved as {submission_csv_path}")

