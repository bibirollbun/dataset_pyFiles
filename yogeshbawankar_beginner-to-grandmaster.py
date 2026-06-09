import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# Let's centralize all our file paths. Good practice.
class CFG:
    TRAIN_PATH = "/kaggle/input/playground-series-s5e8/train.csv"
    TEST_PATH = "/kaggle/input/playground-series-s5e8/test.csv"
    SUBMISSION_PATH = "/kaggle/input/playground-series-s5e8/sample_submission.csv"
    ID_COL = "id"
    TARGET_COL = "y"

config = CFG()


# --- Load Data ---
# Load all the necessary files.
train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)
submission_df = pd.read_csv(config.SUBMISSION_PATH)



# --- Combine for Preprocessing ---
# We combine train and test to ensure identical feature engineering.
# First, let's drop the original target from the training set and the ID.
# We'll save the target variable separately.
target = train_df[config.TARGET_COL]
train_ids = train_df[config.ID_COL]
test_ids = test_df[config.ID_COL]


# Now, drop the columns that won't be used as features.
train_df = train_df.drop(columns=[config.ID_COL, config.TARGET_COL])
test_df = test_df.drop(columns=[config.ID_COL])


# Concatenate them into a single dataframe for processing.
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)


print("--- Data Shapes ---")
print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Combined data shape: {combined_df.shape}")
print("\n--- Combined DataFrame Head ---")
combined_df.head()


# Let's get a high level technical summary of our combined dataset.
print("--- Dataframe Info ---")
combined_df.info()


# Statistical summary for numerical features
print("\n--- Numerical Feature Summary ---")
display(combined_df.describe())

# Summary for categorical features
print("\n--- Categorical Feature Summary ---")
display(combined_df.describe(include='object'))

# Analyze the target variable's distribution
print("\n--- Target Variable Distribution ---")
target_distribution = target.value_counts(normalize=True) * 100
print(target_distribution)

# Visualize the target distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=target)
plt.title('Distribution of Term Deposit Subscription (y)', fontsize=16)
plt.xlabel('Subscribed (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# Let's create a temporary dataframe for plotting
plot_df = train_df.copy()
plot_df['y'] = target

# --- Numerical Features vs. Target ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(x='y', y='duration', data=plot_df, ax=axes[0])
axes[0].set_title('Call Duration vs. Subscription')
sns.boxplot(x='y', y='age', data=plot_df, ax=axes[1])
axes[1].set_title('Client Age vs. Subscription')
plt.show()

# --- Categorical Features vs. Target ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.countplot(x='poutcome', hue='y', data=plot_df, ax=axes[0])
axes[0].set_title('Previous Outcome vs. Subscription')
sns.countplot(x='contact', hue='y', data=plot_df, ax=axes[1])
axes[1].set_title('Contact Method vs. Subscription')
plt.show()


# Take closer look at the key distribution of neumrical features to understand their sahpe and skewness. 

# Create a temporary dataframe for plotting
plot_df = train_df.copy()
plot_df['y'] = target

# Set up the plotting figure
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Distribution of Key Numerical Features by Subscription Outcome', fontsize=20)

# Plot for 'balance'
sns.kdeplot(data=plot_df, x='balance', hue='y', fill=True, common_norm=False, ax=axes[0, 0])
axes[0, 0].set_title('Client Account Balance')
axes[0, 0].set_xlim(-5000, 10000) # Zoom in to see the main distribution

# Plot for 'age'
sns.kdeplot(data=plot_df, x='age', hue='y', fill=True, common_norm=False, ax=axes[0, 1])
axes[0, 1].set_title('Client Age')

# Plot for 'duration'
sns.kdeplot(data=plot_df, x='duration', hue='y', fill=True, common_norm=False, ax=axes[1, 0])
axes[1, 0].set_title('Call Duration')
axes[1, 0].set_xlim(0, 2000) # Zoom in

# Plot for 'campaign' (number of contacts)
sns.kdeplot(data=plot_df, x='campaign', hue='y', fill=True, common_norm=False, ax=axes[1, 1])
axes[1, 1].set_title('Number of Contacts During Campaign')
axes[1, 1].set_xlim(0, 20) # Zoom in

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Let's find out who are the clients that say 'YES' For ex, Job, Education & Marital Status

# Calculate subscription rate by job
job_subscription_rate = plot_df.groupby('job')['y'].mean().sort_values(ascending=False) * 100

# Create the plot
plt.figure(figsize=(12, 7))
sns.barplot(x=job_subscription_rate.index, y=job_subscription_rate.values, palette='viridis')
plt.title('Subscription Rate (%) by Job Title', fontsize=16)
plt.xlabel('Job Title', fontsize=12)
plt.ylabel('Subscription Rate (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 35) # Set y-axis limit to better compare rates
plt.show()


# Bank runs the campaigns over time. Let's see if the campaign is effective in certain months. 

# Define the order of months
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# Calculate subscription rate by month
monthly_rate = plot_df.groupby('month')['y'].mean().reindex(month_order) * 100

# Plotting
plt.figure(figsize=(12, 7))
monthly_rate.plot(kind='bar', color='teal', alpha=0.8)
plt.title('Subscription Rate (%) by Month', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Subscription Rate (%)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# A correlation heatmap is great to see which varibale moves together. 

# --- Correlation Heatmap ---
# Select only the numerical columns for the correlation matrix
numerical_cols = combined_df.select_dtypes(include=np.number).columns

# Calculate the correlation matrix
correlation_matrix = combined_df[numerical_cols].corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)

# Add a title
plt.title('Correlation Matrix of Numerical Features', fontsize=16)

# Show the plot
plt.show()


# STRATEGIC APPROCACH

# --- 1. Feature Engineering (Corrected) ---
# Create a fresh copy to ensure our original combined_df is untouched.
df_processed = combined_df.copy()

print("Transforming features...")

# --- Mapping and Ordinal Encoding ---
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})

education_map = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
df_processed['education'] = df_processed['education'].map(education_map)

# Create a dictionary to translate month abbreviations to numbers.
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
df_processed['month'] = df_processed['month'].map(month_map)

# --- Now, proceed with Cyclical Feature Encoding ---

df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'].astype(int) / 12.0)
df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'].astype(int) / 12.0)
df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day'] / 31.0)
df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day'] / 31.0)
df_processed = df_processed.drop(['month', 'day'], axis=1)

# --- One-Hot Encode the remaining nominal features ---
categorical_cols = ['job', 'marital', 'contact', 'poutcome']
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True, dtype=int)

print("Feature engineering complete.")
print(f"Data shape after processing: {df_processed.shape}")

# Verify the changes by looking at the first few rows.
df_processed.head()


# --- Baseline Modeling with LightGBM ---

# Separate the processed data back into training and testing sets
X = df_processed.iloc[:len(train_df)].copy()
y = target.copy()
X_test = df_processed.iloc[len(train_df):].copy()

# --- Stratified K-Fold Cross-Validation ---
# We use 5 folds, a standard choice for robust validation.
# shuffle=True and random_state ensure the splits are random but reproducible.
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Initialize arrays to store predictions
# 'oof' (out-of-fold) predictions are for the training data, used to calculate our local score.
# 'test_preds' will store the averaged predictions for the final submission.
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("\\nStarting LightGBM Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"===== FOLD {fold+1} =====")
    
    # Split the data for this fold
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Initialize and train the LightGBM model
    # Using default parameters is perfect for a strong baseline.
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict probabilities for the validation set and store them
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    # Predict probabilities for the test set and add them to our running average
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

# --- 3. Evaluate Baseline ---
# Calculate the overall ROC AUC score on our out-of-fold predictions
local_auc = roc_auc_score(y, oof_preds)
print(f"\\n--- Baseline Model Evaluation ---")
print(f"Local ROC AUC Score (LightGBM): {local_auc:.5f}")


# Tactical Approach

# --- Calculating the Class Weight ---
# The standard formula is the count of the majority class ('no') 
# divided by the count of the minority class ('yes').
neg_count = y.value_counts()[0]
pos_count = y.value_counts()[1]
scale_pos_weight_value = neg_count / pos_count

print(f"Negative Class Count: {neg_count}")
print(f"Positive Class Count: {pos_count}")
print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# --- Re-running Cross-Validation with the Weight ---
oof_preds_weighted = np.zeros(len(X))
test_preds_weighted = np.zeros(len(X_test))

print("\\nStarting LightGBM Cross-Validation with Class Weights...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"===== FOLD {fold+1} =====")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Initialize the model WITH the scale_pos_weight parameter
    model_weighted = lgb.LGBMClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight_value
    )
    
    model_weighted.fit(X_train, y_train)
    
    # Store predictions
    val_preds = model_weighted.predict_proba(X_val)[:, 1]
    oof_preds_weighted[val_idx] = val_preds
    test_preds_weighted += model_weighted.predict_proba(X_test)[:, 1] / N_SPLITS

# --- Evaluate the Weighted Model ---
local_auc_weighted = roc_auc_score(y, oof_preds_weighted)
print(f"\\n--- Weighted Model Evaluation ---")
print(f"Local ROC AUC Score (LightGBM):               {local_auc:.5f}")
print(f"Local ROC AUC Score (LightGBM with weights): {local_auc_weighted:.5f}")


# --- Advanced Feature Engineering for CatBoost ---

# Start with a fresh copy of the original combined dataframe.
df_adv_cat = combined_df.copy()

print("Creating advanced interaction and aggregation features...")

# --- 1. Interaction Features ---
# Create new categorical features by combining our strongest predictors.
# CatBoost can handle these new text-based categories directly.
df_adv_cat['poutcome_job'] = df_adv_cat['poutcome'].astype(str) + "_" + df_adv_cat['job'].astype(str)
df_adv_cat['month_contact'] = df_adv_cat['month'].astype(str) + "_" + df_adv_cat['contact'].astype(str)

# --- 2. Aggregation Features ---
# Create features that provide context based on categorical groupings.
# Used .groupby().transform() to calculate statistics for each group.
# and map them back to every row in the dataframe.

# --- Balance-based aggregations ---
# Average balance for each job and education level.
df_adv_cat['job_avg_balance'] = df_adv_cat.groupby('job')['balance'].transform('mean')
df_adv_cat['education_avg_balance'] = df_adv_cat.groupby('education')['balance'].transform('mean')
# Difference from the average
df_adv_cat['balance_vs_job_avg'] = df_adv_cat['balance'] - df_adv_cat['job_avg_balance']

# --- Duration-based aggregations ---
# Average duration for each poutcome and month.
df_adv_cat['poutcome_avg_duration'] = df_adv_cat.groupby('poutcome')['duration'].transform('mean')
df_adv_cat['month_avg_duration'] = df_adv_cat.groupby('month')['duration'].transform('mean')
# Difference from the average
df_adv_cat['duration_vs_poutcome_avg'] = df_adv_cat['duration'] - df_adv_cat['poutcome_avg_duration']

# --- Campaign-based aggregations ---
# Average number of campaign contacts for each job.
df_adv_cat['job_avg_campaign'] = df_adv_cat.groupby('job')['campaign'].transform('mean')
df_adv_cat['campaign_vs_job_avg'] = df_adv_cat['campaign'] - df_adv_cat['job_avg_campaign']


print("Advanced feature engineering complete.")
print(f"Data shape after processing: {df_adv_cat.shape}")


# --- 1. Prepare Data for Modeling ---

# Separate the ADVANCED processed data back into training and testing sets.
X_adv_cat = df_adv_cat.iloc[:len(train_df)].copy()
y_adv_cat = target.copy()
X_test_adv_cat = df_adv_cat.iloc[len(train_df):].copy()

# Identify all columns that should be treated as categorical.
# This now includes our newly created interaction features.
categorical_features_advanced = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 
    'contact', 'month', 'poutcome', 'poutcome_job', 'month_contact'
]

# --- 2. Run CatBoost Cross-Validation ---
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds_adv_cat = np.zeros(len(X_adv_cat))
test_preds_adv_cat = np.zeros(len(X_test_adv_cat))

print("\\nStarting ADVANCED CatBoost Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_adv_cat, y_adv_cat)):
    print(f"===== FOLD {fold+1} =====")
    
    X_train, y_train = X_adv_cat.iloc[train_idx], y_adv_cat.iloc[train_idx]
    X_val, y_val = X_adv_cat.iloc[val_idx], y_adv_cat.iloc[val_idx]

    model_adv_cat = CatBoostClassifier(
        random_state=42,
        cat_features=categorical_features_advanced,
        verbose=0
    )
    
    model_adv_cat.fit(X_train, y_train)
    
    val_preds = model_adv_cat.predict_proba(X_val)[:, 1]
    oof_preds_adv_cat[val_idx] = val_preds
    test_preds_adv_cat += model_adv_cat.predict_proba(X_test_adv_cat)[:, 1] / N_SPLITS

# --- 3. Evaluate the Advanced Model ---
local_auc_adv_cat = roc_auc_score(y_adv_cat, oof_preds_adv_cat)
print(f"\\n--- Model Performance Comparison ---")
print(f"Local ROC AUC Score (Baseline LightGBM):    {local_auc:.5f}") 
print(f"Local ROC AUC Score (Advanced CatBoost):    {local_auc_adv_cat:.5f}")


# --- Create Submission File for Advanced CatBoost ---

# Create a DataFrame with the test IDs and the averaged predictions
catboost_submission_df = pd.DataFrame({'id': test_ids, 'y': test_preds_adv_cat})

# Save the DataFrame to a CSV file in the correct format
catboost_submission_df.to_csv('submission_catboost.csv', index=False)

print("\\nCatBoost submission file 'submission_catboost.csv' created successfully.")

# Display the first few rows to verify the format
catboost_submission_df.head()




