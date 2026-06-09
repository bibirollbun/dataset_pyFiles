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


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


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
    
    # Hyperparameters
    solver = "liblinear"  # Try "saga" for comparison
    max_iter = 1000
    class_weight = "balanced"
    C = 0.5  # Regularization strength (lower = stronger regularization)
    penalty = "l2"  # Try "l1" for feature selection
    tol = 1e-5  # Convergence tolerance


# Configure Pandas to display each DataFrame row in one line without wrapping
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


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


# # Drop Dest column to vrf the efect on thge result
# train_df_no_dest = train_df.drop(columns=["Dest", "Distance"])


# Sample dataset (stratified)
stratify_cols = train_df[["dep_delayed_15min", "UniqueCarrier"]].astype(str)  # Convert to string to avoid numerical issues
if CFG.debug:
    train_sample, _ = train_test_split(
        train_df, 
        test_size = (1 - CFG.sample_fraction), 
        stratify = stratify_cols, 
        random_state = CFG.random_state
    )
else:
    train_sample = train_df.copy()

# Convert target to binary (0 = No Delay, 1 = Delay) without modifying train_df
y_lr = train_sample[CFG.target].map({"N": 0, "Y": 1})


# Define preprocessor pipeline (One-Hot Encoding only for categorical features)
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CFG.cat_features)
], remainder="passthrough")  # Keeps numerical features unchanged

# Apply transformation
X_transformed = preprocessor.fit_transform(train_sample.drop(columns=[CFG.target]))

# Extract feature names correctly
ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CFG.cat_features)
remainder_feature_names = train_sample.drop(columns=[CFG.target] + CFG.cat_features).columns.tolist()

# Combine categorical and numerical features
final_feature_names = list(ohe_feature_names) + remainder_feature_names

# Debug: Print shape comparison
print(f"Shape of X_transformed: {X_transformed.shape}, Expected columns: {len(final_feature_names)}")

# Convert transformed array back to DataFrame
X_lr = pd.DataFrame(X_transformed, columns=final_feature_names)

# Debug: Print DataFrame shape to verify correctness
print(f"Shape of X_lr DataFrame: {X_lr.shape}")


# Split into Train and Temp 
X_train, X_temp, y_train, y_temp = train_test_split(X_lr, y_lr, test_size = (1 - CFG.train_size), stratify = y_lr, random_state = CFG.random_state)

# Split Temp into Valid and Train_Test
X_valid, X_train_test, y_valid, y_train_test = train_test_split(X_temp, y_temp, test_size = CFG.test_valid_rel, stratify = y_temp, random_state = CFG.random_state)

# Print dataset sizes
print(f"Train size: {X_train.shape}, Valid size: {X_valid.shape}, Train_Test size: {X_train_test.shape}")


# Initialize model
logreg = LogisticRegression(
    solver = CFG.solver, 
    max_iter = CFG.max_iter, 
    random_state = CFG.random_state, 
    class_weight = CFG.class_weight,
    C=CFG.C,
    penalty=CFG.penalty,
    tol=CFG.tol
)


# Train model
logreg.fit(X_train, y_train)


# Predictions
y_valid_prob = logreg.predict_proba(X_valid)[:, 1]  # Probability scores for ROC AUC


for cut in CFG.thresholds:
    print(f"\n### Threshold: {cut} ###")
    
    # Convert probabilities to binary predictions
    y_valid_pred = (y_valid_prob >= cut).astype(int)
    
    # Evaluation Metrics
    roc_auc = roc_auc_score(y_valid, y_valid_prob)
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


# Apply the same transformation to the test set
X_test_transformed = preprocessor.transform(test_df)

# Convert transformed array back to DataFrame
X_test_lr = pd.DataFrame(X_test_transformed, columns = final_feature_names)

# Make predictions
y_test_proba = logreg.predict_proba(X_test_lr)[:, 1]  # Get probability of class 1 (delayed)


# Create the submission DataFrame
submission_df = pd.DataFrame({
    "id": range(len(y_test_proba)),
    "dep_delayed_15min": y_test_proba
})

# Save as CSV
submission_csv_path = CFG.submission_file
submission_df.to_csv(submission_csv_path, index=False)


print(f"Submission file saved as {submission_csv_path}")

