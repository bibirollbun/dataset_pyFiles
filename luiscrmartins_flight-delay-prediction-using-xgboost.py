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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import xgboost as xgb
from xgboost import XGBClassifier


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
    n_jobs = -1
    
    thresholds = [0.5, 0.4, 0.3]
    threshold = 0.5
    xgb_params = {
        "n_estimators": 200, 
        "learning_rate": 0.1, 
        "max_depth": 9, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "random_state": random_state,
    }

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9]
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



def check_category_differences(train_df, test_df, categorical_features):
    """
    Compare the unique categories in categorical features between training and test sets.
    
    Args:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Test dataset.
        categorical_features (list): List of categorical columns to compare.

    Returns:
        None. Prints the categories exclusive to train and test.
    """
    for col in categorical_features:
        train_categories = set(train_df[col].unique())
        test_categories = set(test_df[col].unique())

        missing_in_test = train_categories - test_categories
        missing_in_train = test_categories - train_categories

        print(f"Feature: {col}")
        print(f"  Categories in train but missing in test: {missing_in_test}")
        print(f"  Categories in test but missing in train: {missing_in_train}")
        print("-" * 60)

# Run the category comparison
check_category_differences(train_df, test_df, CFG.cat_features)


def preprocess_ohe(train_df, test_df, target_column, cat_features, train_size, test_valid_rel, random_state):
    """
    Preprocess training and test data using One-Hot Encoding (OHE).
    - Concatenates train and test data before applying transformations to ensure consistency.
    - Splits the train set into training, validation, and independent test subsets.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset.
        target_column (str): The target column name.
        cat_features (list): List of categorical features.
        train_size (float): Proportion of data for training.
        test_valid_rel (float): Test size relative to the validation set.
        random_state (int): Random seed for reproducibility.

    Returns:
        - X_train, X_valid, X_train_test, X_test: Processed feature sets.
        - y_train, y_valid, y_train_test: Corresponding labels.

    """

    # Convert target column to binary labels
    y = train_df[target_column].map({"N": 0, "Y": 1})

    # Add identifier column to separate later
    train_df["dataset_type"] = "train"
    test_df["dataset_type"] = "test"

    # Drop target column from train data before concatenation
    train_df_features = train_df.drop(columns=[target_column])
    
    # Concatenate train and test to ensure consistent feature transformations
    combined_df = pd.concat([train_df_features, test_df], axis=0, ignore_index=True)

    # Define One-Hot Encoder for categorical features
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ], remainder="passthrough")

    # Apply transformation
    X_combined_transformed = preprocessor.fit_transform(combined_df.drop(columns=["dataset_type"]))

    # Extract feature names
    ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
    remainder_feature_names = combined_df.drop(columns=cat_features + ["dataset_type"]).columns.tolist()
    final_feature_names = list(ohe_feature_names) + remainder_feature_names

    # Convert transformed data back to DataFrame
    X_combined = pd.DataFrame(X_combined_transformed, columns=final_feature_names)

    # Split train and test based on the identifier
    X_train_full = X_combined[combined_df["dataset_type"] == "train"].reset_index(drop=True)
    X_test = X_combined[combined_df["dataset_type"] == "test"].reset_index(drop=True)

    # Train-validation-test split from training data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_train_full, y, train_size=train_size, stratify=y, random_state=random_state
    )
    X_valid, X_train_test, y_valid, y_train_test = train_test_split(
        X_temp, y_temp, test_size=test_valid_rel, stratify=y_temp, random_state=random_state
    )

    return X_train, X_valid, X_train_test, X_test, y_train, y_valid, y_train_test



def preprocess_le(train_df, cat_features, target, train_size=0.7, test_valid_rel=0.5, random_state=42):
    """
    Preprocessing function using Label Encoding (LE).
    
    Args:
        train_df (pd.DataFrame): The input training dataset.
        cat_features (list): List of categorical features.
        target (str): Target column name.
        train_size (float): Proportion of data for training.
        test_valid_rel (float): Split ratio for validation and test.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    # Convert target to binary
    y = train_df[target].map({"N": 0, "Y": 1})

    # Label Encode categorical features
    X = train_df.drop(columns=[target]).copy()
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=test_valid_rel, stratify=y_temp, random_state=random_state)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def preprocess_le_test(test_df, cat_features, fitted_encoders):
    """
    Apply Label Encoding (LE) preprocessing to the test dataset using the trained label encoders.
    
    Args:
        test_df (pd.DataFrame): The test dataset.
        cat_features (list): List of categorical features.
        fitted_encoders (dict): A dictionary of fitted LabelEncoders for each categorical feature.
    
    Returns:
        X_test (pd.DataFrame): Transformed test data as DataFrame.
    """
    X_test = test_df.copy()
    
    # Apply the saved encoders to categorical features
    for col in cat_features:
        if col in fitted_encoders:
            X_test[col] = fitted_encoders[col].transform(X_test[col])
    
    return X_test


print(train_df.columns)


# One-Hot Encoding:
X_train, X_valid, X_train_test, X_test, y_train, y_valid, y_train_test = preprocess_ohe(
    train_df = train_df, 
    test_df = test_df,
    target_column = CFG.target, 
    cat_features = CFG.cat_features, 
    train_size = CFG.train_size, 
    test_valid_rel = CFG.test_valid_rel, 
    random_state = CFG.random_state
)


xgb_model_ohe = XGBClassifier(**CFG.xgb_params)
xgb_model_ohe.fit(X_train, y_train)


y_pred_proba = xgb_model_ohe.predict_proba(X_valid)[:, 1]
roc_auc = roc_auc_score(y_valid, y_pred_proba)
print(f"ROC AUC: {roc_auc:.4f}")


y_pred_test_proba = xgb_model_ohe.predict_proba(X_train_test)[:, 1]
roc_auc_test = roc_auc_score(y_train_test, y_pred_test_proba)
print(f"ROC AUC: {roc_auc_test:.4f}")


y_pred_test_proba


print(CFG.xgb_params)


# Generate predictions
y_test_proba = xgb_model_ohe.predict_proba(X_test)[:, 1]


# Create the submission DataFrame
submission = pd.DataFrame({"id": test_df.index, "dep_delayed_15min": y_test_proba})


# Save as CSV
submission.to_csv(CFG.submission_file, index=False)


print(submission)


submission.describe()


import matplotlib.pyplot as plt
plt.hist(y_test_proba, bins=50)
plt.title("Distribution of Predicted Probabilities")
plt.show()


train_df["dep_delayed_15min"].value_counts(normalize=True)


