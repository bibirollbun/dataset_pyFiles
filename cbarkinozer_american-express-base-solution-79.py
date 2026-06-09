!pip install numpy pandas seaborn matplotlib scikit-learn lightgbm


import pandas as pd

# === STEP 1: CREATE A MANAGEABLE DATA SAMPLE ===

# We'll use the file paths you confirmed in the previous cell.
TRAIN_DATA_PATH = 'train_data.csv'
TRAIN_LABELS_PATH = 'train_labels.csv'

# Read a fraction of the main data file to avoid memory errors.
# 1 million rows is a good starting point for EDA and pipeline development.
print("Reading a sample of the training data")
df_train_sample = pd.read_csv(TRAIN_DATA_PATH, nrows=1000000) # use limit if you have limited resources nrows=1000000

# Load the training labels. This file is small, so we can load it fully.
print("Reading training labels...")
df_train_labels = pd.read_csv(TRAIN_LABELS_PATH)

# --- Merge the sample with labels ---
# Find out which unique customers are present in our 1M row sample.
customers_in_sample = df_train_sample['customer_ID'].unique()

# Filter the labels dataframe to only include the customers from our sample.
df_labels_sample = df_train_labels[df_train_labels['customer_ID'].isin(customers_in_sample)]

# Merge the labels into our main training dataframe.
# We use a left merge to ensure we keep all rows from our data sample and add the target column.
print("Merging data sample with labels...")
df = pd.merge(df_train_sample, df_labels_sample, on='customer_ID', how='left')

# --- Verification ---
print("\n--- Verification of the Sampled DataFrame ---")
print(f"Shape of our final sample dataframe: {df.shape}")
print(f"Number of unique customers in the sample: {df['customer_ID'].nunique()}")
print(f"Number of rows with a null target value: {df['target'].isnull().sum()} (should be 0)")

print("\nSample DataFrame Head:")
df.head()

print("\nSample DataFrame Info:")
df.info()


import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style for better visualization
sns.set_style('whitegrid')

print("--- EDA Step 1: DataFrame Numerical Summary ---")
# .describe() gives us a statistical summary of the numerical columns.
df.describe()

print("\n--- EDA Step 2: Missing Values Analysis ---")
missing_values_perc = (df.isnull().sum() / len(df)) * 100
print("Percentage of missing values for each column (Top 20):")
# We display the top 20 columns with the most missing values.
missing_values_perc.sort_values(ascending=False).head(20)

print("\n--- EDA Step 3: Target Variable Imbalance Check ---")
# IMPORTANT: Since each customer has multiple rows, we should check the target distribution
# on a per-customer basis to get the true picture. We do this by dropping duplicates.
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df.drop_duplicates(subset=['customer_ID']))
plt.title('Distribution of Target Variable (Per Unique Customer)')
plt.xlabel('Default (1) vs. No Default (0)')
plt.ylabel('Number of Unique Customers')
# Adding percentage labels
unique_customers = df.drop_duplicates(subset=['customer_ID'])
total = len(unique_customers)
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().text(p.get_x() + p.get_width()/2., height + 5, f'{100*height/total:.2f}%', ha='center')
plt.show()


print("\n--- EDA Step 4: Histograms for Key Numerical Features ---")
# Let's look at the distribution of a Payment, a Balance, and a Delinquency variable.
key_num_features = ['P_2', 'B_1', 'D_39']
df[key_num_features].hist(bins=50, figsize=(18, 5), layout=(1, 3))
plt.suptitle('Distribution of Key Numerical Features', size=16, y=1.02)
plt.show()

print("\n--- EDA Step 5: Count Plots for Key Categorical Features ---")
# These were listed in the competition's data description.
key_cat_features = ['B_30', 'B_38', 'D_63', 'D_64', 'D_68']
for col in key_cat_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f'Count Plot for {col}')
    plt.xscale('log') # Use a log scale if counts are highly skewed
    plt.show()


# === STEP 3: FEATURE ENGINEERING (AGGREGATION) ===

print("Starting feature aggregation. This may take a minute...")

# Identify categorical and numerical features
# Exclude customer_ID, S_2 (date), and target from the feature list
features = [col for col in df.columns if col not in ['customer_ID', 'S_2', 'target']]

# D_63, D_64 are categorical as per data description. Let's find others.
cat_features = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126",
    "D_63", "D_64", "D_66", "D_68"
]
# Filter for cat_features that are actually in our dataframe
cat_features = [f for f in cat_features if f in df.columns]

num_features = [f for f in features if f not in cat_features]

# Define aggregations
# For numerical features: get the mean, std, min, max, and last value
num_aggs = {col: ['mean', 'std', 'min', 'max', 'last'] for col in num_features}
# For categorical features: count unique values, get the last value, and the most frequent
cat_aggs = {col: ['count', 'last', 'nunique'] for col in cat_features}

# Combine aggregation dictionaries
all_aggs = {**num_aggs, **cat_aggs}

# Perform the aggregation
df_agg = df.groupby('customer_ID').agg(all_aggs)

# The aggregation creates multi-level columns (e.g., ('P_2', 'mean')).
# Let's flatten them into single-level column names (e.g., 'P_2_mean').
df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
df_agg.reset_index(inplace=True)

# Merge with labels to get the target variable
# We need a dataframe with one row per customer for the labels
df_labels_unique = df.groupby('customer_ID')['target'].first().reset_index()
df_agg = pd.merge(df_agg, df_labels_unique, on='customer_ID', how='left')


# --- Verification ---
print("\n--- Verification of the Aggregated DataFrame ---")
print(f"Shape of our new aggregated dataframe: {df_agg.shape}")
print(f"Number of unique customers (rows): {df_agg.shape[0]}")
print(f"Number of features: {df_agg.shape[1] - 2}") # Subtract customer_ID and target
print(f"Number of nulls in target variable: {df_agg['target'].isnull().sum()} (should be 0)")
print("\nAggregated DataFrame Head:")
df_agg.head()


# === STEP 4: CREATE A BASELINE MODEL ===

import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --- 1. Prepare Data ---
print("Preparing data for the baseline model...")

# Separate features (X) and target (y)
X = df_agg.drop(columns=['customer_ID', 'target'])
y = df_agg['target']

# Identify numerical and categorical feature names from the new aggregated dataframe
# Note: Categorical features now have suffixes like '_last', '_nunique', etc.
cat_features_agg = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype == 'category']
num_features_agg = [col for col in X.columns if col not in cat_features_agg]

print(f"Number of numerical features: {len(num_features_agg)}")
print(f"Number of categorical features: {len(cat_features_agg)}")


# --- 2. Create Preprocessing Pipelines ---
# This pipeline handles missing values and scales the data.
# For numerical data: impute with the median, then scale.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For categorical data: impute with the most frequent value, then one-hot encode.
# handle_unknown='ignore' is important for when the test set has categories not seen in training.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Use ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features_agg),
        ('cat', categorical_transformer, cat_features_agg)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# --- 3. Create the Full Model Pipeline ---
# This chains the preprocessing steps with the final classifier.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# --- 4. Evaluate the Model with Cross-Validation ---
print("\nStarting cross-validation for the baseline model...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Fit the pipeline on the training data for this fold
    model_pipeline.fit(X_train, y_train)

    # Predict probabilities for the validation set
    # We need predict_proba for ROC AUC score
    y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]

    # Calculate and store the AUC score
    fold_auc = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

    # Clean up memory
    del X_train, X_val, y_train, y_val
    gc.collect()

print("\n--- Baseline Model Evaluation ---")
print(f"Average ROC AUC Score: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")


# === STEP 5 (CORRECTED): IMPROVE WITH AN ADVANCED MODEL (LIGHTGBM) ===

import lightgbm as lgb
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

# Suppress LightGBM warnings about no positive gain, which are common on small datasets
warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain: -inf")

# --- 1. Fit the Preprocessor ONCE on the entire dataset ---
print("Fitting the preprocessor on the entire training set...")
# This ensures the OneHotEncoder learns ALL possible categories from the start.
X_transformed = preprocessor.fit_transform(X)
print("Data has been preprocessed.")

# Get the feature names after one-hot encoding
# This is a bit complex but useful for the feature importance plot later
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features_agg)
# Combine with numerical feature names
all_feature_names = num_features_agg + ohe_feature_names.tolist()


# --- 2. Evaluate the Advanced Model with Cross-Validation ---
# Now we will cross-validate on the PRE-TRANSFORMED data.
print("\nStarting cross-validation for the LightGBM model...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgbm_auc_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(X_transformed, y)):
    # Use the pre-transformed data and slice it
    X_train, X_val = X_transformed[train_index], X_transformed[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Create and train the model directly (no pipeline needed inside the loop)
    model = lgb.LGBMClassifier(random_state=42, is_unbalance=True)
    model.fit(X_train, y_train, feature_name=all_feature_names) # Pass feature names for clarity

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate and store AUC
    fold_auc = roc_auc_score(y_val, y_pred_proba)
    lgbm_auc_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

    del X_train, X_val, y_train, y_val, model
    gc.collect()

print("\n--- LightGBM Model Evaluation ---")
baseline_score = np.mean(auc_scores) # auc_scores is from your previous step
lgbm_score = np.mean(lgbm_auc_scores)
print(f"Baseline Logistic Regression Average AUC: {baseline_score:.4f}")
print(f"LightGBM Average ROC AUC Score: {lgbm_score:.4f} (+/- {np.std(lgbm_auc_scores):.4f})")

improvement = lgbm_score - baseline_score
print(f"\nImprovement over baseline: {improvement:.4f}")
if improvement > 0:
    print("Success! The LightGBM model outperformed the baseline.")
else:
    print("The LightGBM model did not outperform the baseline. Further tuning may be required.")


# STEP 6: FEATURE IMPORTANCE AND STORYTELLING

import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Train a Final Model on ALL Data ---
# To get the most reliable feature importances, we'll train one final
# model on the entire preprocessed dataset (X_transformed, y).
print("Training final LightGBM model on all data to get feature importances...")

final_model = lgb.LGBMClassifier(random_state=42, is_unbalance=True)

# We can reuse the preprocessed data and feature names from the previous step
final_model.fit(X_transformed, y, feature_name=all_feature_names)

print("Final model trained.")

# --- 2. Generate and Plot Feature Importances ---
# Create a dataframe of features and their importance scores
feature_importances = pd.DataFrame({
    'feature': all_feature_names,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

# Display the top 20 most important features
print("\nTop 20 Most Important Features:")
feature_importances.head(20)

# Plot the top 20 features
plt.figure(figsize=(12, 8))
sns.barplot(
    x='importance',
    y='feature',
    data=feature_importances.head(20),
    palette='viridis'
)
plt.title('Top 20 Feature Importances from LightGBM Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# --- 3. Draft the Project Summary ---
# Extract the top 3 features for our summary
top_3_features = feature_importances['feature'].head(3).tolist()

print("\n\n--- DRAFT PROJECT SUMMARY ---")
summary = f"""
Our solution successfully predicts customer default by transforming complex, time-series statement data into a powerful set of summary features for each customer.

Key Strategy: We aggregated customer data over time, focusing on the mean, standard deviation, and most recent values of their financial behaviors. This created a rich, single-view dataset for each individual.

Model Performance: A LightGBM classifier proved highly effective, achieving an average ROC AUC score of {lgbm_score:.4f}. This significantly outperformed our robust Logistic Regression baseline score of {baseline_score:.4f}, confirming the presence of complex, non-linear patterns in the data.

Key Predictive Features: The model's decisions are primarily driven by key features such as **{top_3_features[0]}**, **{top_3_features[1]}**, and **{top_3_features[2]}**. This indicates that [Your interpretation here - e.g., 'a customer's most recent payment behavior and the minimum balance they've held are critical indicators of risk.'].

Conclusion: This approach allows for the effective identification of high-risk customers, enabling proactive risk management.
"""
print(summary)


# PRIORITY #1: IMPLEMENT THE OFFICIAL AMEX METRIC

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """
    The official evaluation metric for the American Express Default Prediction competition.
    
    This function is a Python implementation of the metric provided by the competition hosts.
    """
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['rank'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['rank'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
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


# ENGINEERING "RECENCY" FEATURES
# Based on feature importance plot, *_last features are the most powerful.
# Our hypothesis is that the change from a customer's average behavior to their most recent behavior is a very strong signal of changing risk.
# This cell creates those new features.
print("Engineering new 'recency' features based on the feature importance plot...")

# We will use the original aggregated dataframe 'df_agg' from Step 3
# and add new columns to it.

# Let's select the base names of the top features you found
top_features_base = ['P_2', 'D_39', 'B_4', 'B_5', 'B_1', 'B_3', 'D_46', 'S_3', 'D_43', 'B_2']
new_feature_names = []

for col in top_features_base:
    # Check if both 'last' and 'mean' columns exist for this feature
    if f'{col}_last' in df_agg.columns and f'{col}_mean' in df_agg.columns:
        new_feature_name = f'{col}_last_minus_mean'
        df_agg[new_feature_name] = df_agg[f'{col}_last'] - df_agg[f'{col}_mean']
        new_feature_names.append(new_feature_name)

print(f"Successfully created {len(new_feature_names)} new features.")
print("Example of the new features:")
df_agg[['customer_ID'] + new_feature_names].head()


# STEP 5 - RE-EVALUATING WITH OFFICIAL METRIC

import lightgbm as lgb
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain: -inf")

# --- 1. Prepare Data (using the dataframe with our NEW features) ---
print("Preparing data with the newly engineered features...")
X = df_agg.drop(columns=['customer_ID', 'target'])
y = df_agg['target']

# We need to re-identify the feature types since we added new numerical columns
cat_features_agg = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype == 'category']
num_features_agg = [col for col in X.columns if col not in cat_features_agg]

# The preprocessor is the same, but it will be refit on the new data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features_agg),
        ('cat', categorical_transformer, cat_features_agg)
    ],
    remainder='passthrough'
)

# --- 2. Fit the Preprocessor ONCE on the entire NEW dataset ---
print("Fitting the preprocessor on the updated training set...")
X_transformed = preprocessor.fit_transform(X)
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features_agg)
all_feature_names = num_features_agg + ohe_feature_names.tolist()
print("Data has been preprocessed.")

# --- 3. Evaluate with Cross-Validation ---
print("\nStarting cross-validation with BOTH AUC and official Amex Metric...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgbm_auc_scores = []
lgbm_amex_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(X_transformed, y)):
    X_train, X_val = X_transformed[train_index], X_transformed[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMClassifier(random_state=42, is_unbalance=True)
    model.fit(X_train, y_train, feature_name=all_feature_names)

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate AUC
    fold_auc = roc_auc_score(y_val, y_pred_proba)
    lgbm_auc_scores.append(fold_auc)

    # Calculate Amex Metric
    y_val_df = pd.DataFrame(y_val.values, columns=['target'])
    y_pred_df = pd.DataFrame(y_pred_proba, columns=['prediction'])
    fold_amex = amex_metric(y_val_df, y_pred_df)
    lgbm_amex_scores.append(fold_amex)
    
    print(f"Fold {fold+1} -> AUC: {fold_auc:.4f}, Amex Metric: {fold_amex:.4f}")
    
    del model, X_train, X_val, y_train, y_val, y_val_df, y_pred_df
    gc.collect()

print("\n--- Final LightGBM Model Evaluation (with new features) ---")
print(f"LightGBM Average ROC AUC Score: {np.mean(lgbm_auc_scores):.4f}")
print(f"LightGBM Average Amex Metric:  {np.mean(lgbm_amex_scores):.4f} (+/- {np.std(lgbm_amex_scores):.4f})")


# PART 1 - CREATING FIRST SUBMISSION FILE

import pandas as pd
import numpy as np

print("Starting the submission process...")

# --- 1. Train the Final Model on ALL Training Data ---
# We use the final preprocessed training data from the previous step (X_transformed, y)
# and the final list of feature names (all_feature_names)

print("Training the final LightGBM model on the entire 1M-row training sample...")
final_model = lgb.LGBMClassifier(random_state=42, is_unbalance=True)
final_model.fit(X_transformed, y, feature_name=all_feature_names)
print("Final model has been trained.")


# --- 2. Process the Test Data ---
# The test data is also large, so we need a memory-efficient way to process it.
# We will process it customer by customer.
print("Loading and processing the test data...")
TEST_DATA_PATH = '/kaggle/input/amex-default-prediction/test_data.csv'

# This is a simplified chunking-like process for aggregation
# A more optimized version would read the file in chunks, but this works for demonstration
test_df = pd.read_csv(TEST_DATA_PATH)

# Perform the SAME aggregation as on the training data
print("Aggregating test data features...")
test_agg = test_df.groupby('customer_ID').agg(all_aggs) # all_aggs was defined in your Step 3
test_agg.columns = ['_'.join(col).strip() for col in test_agg.columns.values]
test_agg.reset_index(inplace=True)

# Perform the SAME feature engineering for "recency" features
print("Engineering 'recency' features for test data...")
for col in top_features_base: # top_features_base was defined in your "recency" cell
    if f'{col}_last' in test_agg.columns and f'{col}_mean' in test_agg.columns:
        test_agg[f'{col}_last_minus_mean'] = test_agg[f'{col}_last'] - test_agg[f'{col}_mean']

# Keep track of customer IDs for the submission file
test_customer_ids = test_agg['customer_ID']
X_test = test_agg.drop(columns=['customer_ID'])

# --- 3. Make Predictions ---
# Use the SAME preprocessor that was FIT on the TRAINING data.
# This is crucial to avoid data leakage!
print("Preprocessing test data and making predictions...")
X_test_transformed = preprocessor.transform(X_test)
test_predictions = final_model.predict_proba(X_test_transformed)[:, 1]

# --- 4. Create Submission File ---
print("Creating submission file...")
submission_df = pd.DataFrame({
    'customer_ID': test_customer_ids,
    'prediction': test_predictions
})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("You can now go to the 'Output' section of your notebook on the right, find the file, and click 'Submit'.")
display(submission_df.head())


# === CELL 1: CONFIGURATION, SETUP, AND METRIC ===

import os
import gc
import warnings
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration Class ---
class CFG:
    # Correct data paths to the original CSV files
    TRAIN_PATH = 'train_data.csv'
    TEST_PATH = 'test_data.csv'
    TRAIN_LABELS_PATH = 'train_labels.csv'
    SUBMISSION_PATH = 'sample_submission.csv'
    
    # Seeds for blending
    seeds = [42, 1337, 2025]
    
    # Model settings
    n_folds = 4
    target = 'target'
    
    # LightGBM parameters (tuned for performance)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'dart',
        'seed': 42,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
    }

# --- Function to seed everything for reproducibility ---
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- Official Amex Metric ---
def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)

def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True

print("Configuration and setup complete.")


# CELL 2: DATA PROCESSING & FEATURE ENGINEERING (FOR TRAINING ONLY)

# Define feature groups
cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
num_features = [col for col in pd.read_csv(CFG.TRAIN_PATH, nrows=1).columns if col not in ['customer_ID', 'S_2'] + cat_features]

def process_and_feature_engineer_df(df):
    # This function processes a dataframe (either a chunk or a full df)
    
    # 1. Basic Aggregations
    num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    num_agg.columns = ['_'.join(x) for x in num_agg.columns]
    cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    cat_agg.columns = ['_'.join(x) for x in cat_agg.columns]
    
    # 2. Momentum Features (_diff1)
    def get_difference(data, num_features):
        df_diff = []
        customer_ids = []
        for customer_id, group in data.groupby(['customer_ID']):
            diff_df = group[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
            df_diff.append(diff_df)
            customer_ids.append(customer_id)
        df_diff = np.concatenate(df_diff, axis=0)
        df_diff = pd.DataFrame(df_diff, columns=[col + '_diff1' for col in num_features])
        df_diff['customer_ID'] = customer_ids
        return df_diff
        
    diff_feats = get_difference(df, num_features)
    
    # Combine all features
    df_agg = pd.concat([num_agg, cat_agg], axis=1).reset_index()
    df_agg = df_agg.merge(diff_feats, on='customer_ID', how='left')
    
    # 3. Recency vs. Average Features (_last_minus_mean)
    print("Creating recency vs. average features...")
    num_feature_names_agg = [f for f in num_features]
    for col in tqdm(num_feature_names_agg):
        if f'{col}_last' in df_agg.columns and f'{col}_mean' in df_agg.columns:
            df_agg[f'{col}_last_minus_mean'] = df_agg[f'{col}_last'] - df_agg[f'{col}_mean']
            
    return df_agg

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns, desc="Reducing Memory"):
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage decreased from {start_mem:.2f}MB to {end_mem:.2f}MB')
    return df

# --- Execute Processing on a 1M row sample of Training Data ---
print("Processing Train Data Sample (1,000,000 rows)...")
train_df_sample = pd.read_csv(CFG.TRAIN_PATH, nrows=1000000)
train = process_and_feature_engineer_df(train_df_sample)

train_labels = pd.read_csv(CFG.TRAIN_LABELS_PATH)
train = train.merge(train_labels, on='customer_ID', how='left')
train = reduce_mem_usage(train)

del train_df_sample, train_labels; gc.collect()
print("\nTraining data prepared.")


# CELL 3: MODEL TRAINING

# Import the necessary callbacks from lightgbm
from lightgbm import early_stopping, log_evaluation

# We need to pre-fit the LabelEncoders on the training data so we can reuse them on test chunks
cat_features_last = [f"{cf}_last" for cf in cat_features]
encoders = {}
for cat_col in tqdm(cat_features_last, desc="Fitting LabelEncoders"):
    encoder = LabelEncoder()
    train[cat_col] = train[cat_col].fillna('NaN').astype(str)
    train[cat_col] = encoder.fit_transform(train[cat_col])
    encoders[cat_col] = encoder # Save the fitted encoder

# Get feature list
features = [col for col in train.columns if col not in ['customer_ID', CFG.target]]

# To store Out-of-Fold predictions and models
oof_predictions = np.zeros(len(train))
trained_models = []

# --- SEED BLENDING LOOP ---
for seed in CFG.seeds:
    print("\n" + "="*50)
    print(f"TRAINING WITH SEED: {seed}")
    print("="*50)
    
    seed_everything(seed)
    CFG.params['seed'] = seed
    
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=seed)
    
    # --- FOLD LOOP ---
    for fold, (trn_ind, val_ind) in enumerate(skf.split(train, train[CFG.target])):
        print(f"\n----------- Fold {fold+1} -----------")
        
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[CFG.target].iloc[trn_ind], train[CFG.target].iloc[val_ind]
        
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=cat_features_last)
        lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature=cat_features_last)
        
        # --- THIS IS THE CORRECTED PART ---
        model = lgb.train(
            params=CFG.params,
            train_set=lgb_train,
            num_boost_round=10000,
            valid_sets=[lgb_train, lgb_valid],
            callbacks=[ # Use the callbacks parameter instead
                early_stopping(stopping_rounds=500),
                log_evaluation(period=500)
            ],
            feval=lgb_amex_metric
        )
        # --- END OF CORRECTION ---
        
        # Save the trained model
        trained_models.append(model)
        
        # Store OOF predictions
        val_pred = model.predict(x_val)
        oof_predictions[val_ind] += val_pred / len(CFG.seeds)
        
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()

# Final CV score based on our training sample
final_cv_score = amex_metric(train[CFG.target], oof_predictions)
print("\n" + "#"*50)
print(f"OVERALL CV SCORE (on 1M sample): {final_cv_score:.5f}")
print(f"Total models trained: {len(trained_models)}")
print("#"*50)


# HELPER FUNCTION: MEMORY REDUCTION
# This function iterates through columns of a dataframe and downcasts numeric types
# to the smallest possible type, drastically reducing memory usage.

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    # float16 can be imprecise, float32 is often a better choice
                    df[col] = df[col].astype(np.float32) 
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df


# CELL 4: CHUNKED INFERENCE & SUBMISSION (OPTIMIZED)

def inference_and_submit_in_chunks(test_path, models, encoders, features, chunksize=500000):
    print(f"Starting chunked inference on {test_path} with chunksize={chunksize}")
    
    # Get all unique customer IDs from the test set first for the final submission df
    print("Getting unique customer IDs from test set...")
    all_test_customer_ids = pd.read_csv(test_path, usecols=['customer_ID'])['customer_ID'].unique()
    
    # Create a placeholder for predictions
    predictions_df = pd.DataFrame(index=all_test_customer_ids, columns=['prediction'], dtype=np.float32)

    # Create an iterator to read the test data in chunks
    chunk_iter = pd.read_csv(test_path, chunksize=chunksize)
    
    # tqdm now needs the total number of chunks to give a better progress bar
    num_rows = 11363762 # Total rows in test_data.csv for AMEX
    total_chunks = (num_rows // chunksize) + 1
    
    for i, test_chunk in tqdm(enumerate(chunk_iter), total=total_chunks, desc="Processing Chunks"):
        print(f"\n--- Chunk {i+1}/{total_chunks} ---")
        
        # We use a try...finally block to ensure memory is always released
        try:
            # 1. Feature Engineer the chunk
            print("Step 1/5: Feature Engineering...")
            test_agg_chunk = process_and_feature_engineer_df(test_chunk)
            
            # 1b. CRITICAL: Reduce memory after creating features
            print("Step 2/5: Reducing memory usage...")
            test_agg_chunk = reduce_mem_usage(test_agg_chunk, verbose=False) # verbose=False to keep logs clean
            
            # 2. Label Encode using the FITTED encoders from training
            print("Step 3/5: Label Encoding...")
            for cat_col, encoder in encoders.items():
                test_agg_chunk[cat_col] = test_agg_chunk[cat_col].fillna('NaN').astype(str)
                unseen_labels = [label for label in test_agg_chunk[cat_col].unique() if label not in encoder.classes_]
                if unseen_labels:
                    encoder.classes_ = np.append(encoder.classes_, unseen_labels)
                test_agg_chunk[cat_col] = encoder.transform(test_agg_chunk[cat_col])
            
            X_test_chunk = test_agg_chunk[features].copy()
            
            # 3. Make Predictions with all models
            print("Step 4/5: Making predictions...")
            chunk_preds = np.zeros(len(X_test_chunk))
            for model in models:
                chunk_preds += model.predict(X_test_chunk) / len(models)
            
            # 4. Store predictions
            print("Step 5/5: Storing predictions...")
            customer_ids_in_chunk = test_agg_chunk['customer_ID'].values
            predictions_df.loc[customer_ids_in_chunk, 'prediction'] = chunk_preds

        finally:
            # This block will run even if an error occurs, preventing memory leaks
            del test_chunk, test_agg_chunk, X_test_chunk, chunk_preds
            gc.collect()

    print("\nAll chunks processed successfully.")
    predictions_df = predictions_df.reset_index().rename(columns={'index': 'customer_ID'})
    return predictions_df

# --- Run Inference and Create Submission File ---
# We use a smaller chunksize to prevent the kernel from crashing due to memory limits.
# 500,000 is a safe starting point. You can try increasing it if it runs too slowly,
# or decreasing it if you still run out of memory.
submission = inference_and_submit_in_chunks(
    test_path=CFG.TEST_PATH,
    models=trained_models,
    encoders=encoders,
    features=features,
    chunksize=500000  # ADJUST THIS VALUE BASED ON YOUR KERNEL'S MEMORY
)

submission.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
submission.head()

