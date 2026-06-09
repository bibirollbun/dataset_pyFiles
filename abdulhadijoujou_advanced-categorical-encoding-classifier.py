# --- Setup and Libraries ---
print("--- Step 0: Installing CatBoost Library ---")
!pip install catboost --quiet
print("--- CatBoost installed successfully ---")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import warnings

# Display settings and warning suppression
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 100) # To display more columns when printing
print("--- Libraries loaded and settings configured ---")
# --- 1. Quick Look at the Data (Loading Data) ---
print("--- 1. Quick Look at the Data (Loading Data) ---")
BASE_PATH = '/kaggle/input/cat-in-the-dat-ii/'

try:
    train_df = pd.read_csv(BASE_PATH + 'train.csv')
    test_df = pd.read_csv(BASE_PATH + 'test.csv')
except FileNotFoundError:
    print("â�Œ Error: Make sure to add 'cat-in-the-dat-ii' data to the notebook.")
    raise

# Separate variables
y = train_df['target']
test_ids = test_df['id']
train_df = train_df.drop(['id', 'target'], axis=1)
test_df = test_df.drop('id', axis=1)

print(f"âœ… Data loaded successfully.")
print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


print("First 5 rows of the training data:")
train_df.head()


train_df.info()


train_df.describe()


train_df.describe(include='O')


train_df.isnull().sum()


train_df.duplicated().sum()


# --- 1. Calculation (Fixes NameError) ---
# Calculate the absolute counts of each target class (0 and 1)
target_counts = y.value_counts()
total_count = len(y)
# Calculate the percentage distribution
target_percentage = (target_counts / total_count) * 100

print("--- Target (y) Distribution Counts ---")
print(target_counts)
print("\n--- Target Class Percentage ---")
print(target_percentage.round(2).astype(str) + '%')

# --- 2. Visualization (Countplot) ---
plt.figure(figsize=(6, 4))
# Note: The 'palette' uses distinct colors for clear visualization.
sns.countplot(x=y, palette=['#1f77b4', '#ff7f0e']) 
plt.title('Target Variable Distribution (Target: 0 and 1)', fontsize=14)
plt.xlabel('Target Value')
plt.ylabel('Count')

# Adding the percentage directly above the bars
for i, count in enumerate(target_counts):
    percentage = target_percentage.iloc[i]
    # Placing the text slightly above the bar (count + 1000) for visibility.
    plt.text(i, count + 1000, f'{percentage:.2f}%', 
             ha='center', va='bottom', fontsize=12)

plt.show()


# --- 1. Data Loading and Initial Setup ---
BASE_PATH = '/kaggle/input/cat-in-the-dat-ii/'

try:
    # Load and drop unnecessary ID/Target columns immediately
    train_df = pd.read_csv(BASE_PATH + 'train.csv').drop(['id', 'target'], axis=1, errors='ignore')
    test_df = pd.read_csv(BASE_PATH + 'test.csv').drop('id', axis=1, errors='ignore')
except FileNotFoundError:
    print("Error: Ensure data is accessible at the specified path.")
    raise

# Concatenate data for combined analysis
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# --- 2. Calculate and Filter Missing Data ---
# Calculate the percentage of missing values for all columns
missing_percentage = full_df.isnull().mean() * 100

# Filter to keep only columns with missing data and sort them
missing_data = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

# --- 3. Visualization and Output ---
if not missing_data.empty:
    plt.figure(figsize=(10, 8))
    
    # Plotting the missing percentage for all relevant features
    sns.barplot(x=missing_data.values, y=missing_data.index, palette="viridis")

    plt.title('Percentage of Missing Values Per Feature', fontsize=16)
    plt.xlabel('Missing Percentage (%)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(axis='x', alpha=0.6) # Added slight transparency to the grid
    plt.tight_layout()
    plt.show()
    
    print("\n--- Top Missing Features ---")
    # Display the top features and their missing percentage
    print(missing_data.head(10).apply(lambda x: f"{x:.2f}%").to_string())
else:
    print("No missing values found in the combined dataset.")


# Assuming 'train_eda' (full original train_df) is available

# Recalculate train_eda just to be safe (if run standalone)
BASE_PATH = '/kaggle/input/cat-in-the-dat-ii/'
train_eda = pd.read_csv(BASE_PATH + 'train.csv')


print("--- 3. Ordinal Features (ord_2) vs. Target Rate ---")

# Calculate P(Target=1) for each category in ord_2
target_rate_ord2 = train_eda.groupby('ord_2')['target'].mean().sort_values()

plt.figure(figsize=(8, 5))
# Plot the mean of the target, which is the probability P(Target=1)
target_rate_ord2.plot(kind='bar', color='darkgreen')
plt.title('Target Rate by Category: ord_2 (P(Target=1))', fontsize=14)
plt.xlabel('ord_2 Categories (Ordered)')
plt.ylabel('Probability of Target=1')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.5)
plt.show()


# Assuming 'train_df' is available (features only)

print("--- 4. Nominal Feature Cardinality ---")

# Select only nominal columns for cardinality check
nominal_cols = [col for col in train_df.columns if 'nom_' in col]
cardinality = train_df[nominal_cols].nunique().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=cardinality.index, y=cardinality.values, palette="rocket")
plt.title('Unique Values Count (Cardinality) for Nominal Features', fontsize=16)
plt.xlabel('Nominal Features')
plt.ylabel('Number of Unique Values')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.6)
plt.tight_layout()
plt.show()


# Assuming 'train_eda' (full original train_df) is available

# Recalculate train_eda just to be safe
BASE_PATH = '/kaggle/input/cat-in-the-dat-ii/'
train_eda = pd.read_csv(BASE_PATH + 'train.csv')


print("--- 5. Cyclical Feature Impact (Monthly Rate) ---")

# Calculate the mean target rate for each month
monthly_target_rate = train_eda.groupby('month')['target'].mean()

plt.figure(figsize=(8, 5))
# Use a line plot to show the cyclical nature
monthly_target_rate.plot(kind='line', marker='o', color='darkblue')
plt.title('Average P(Target=1) Rate Across Months', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Probability of Target=1')
plt.xticks(range(1, 13)) # Ensure all months (1-12) are labeled
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# --- 3. Feature Engineering ---
print("\n--- 3. Feature Engineering ---")

# Concatenate data
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)



# 3.1: Handling Missing Values and Creating Indicators
print("--- 3.1: Handling Missing Values and Creating Indicators ---")
for col in full_df.columns:
    if full_df[col].isnull().any():
        full_df[f'{col}_was_missing'] = full_df[col].isnull().astype(int)
        
        if pd.api.types.is_numeric_dtype(full_df[col]):
            full_df[col] = full_df[col].fillna(-1)  
        else:
            full_df[col] = full_df[col].fillna('NONE')


# 3.2: Ordinal Feature Encoding (ord_*)
print("--- 3.2: Ordinal Feature Encoding (ord_*) ---")
ord_1_mapping = {'Novice': 1, 'Contributor': 2, 'Expert': 3, 'Master': 4, 'Grandmaster': 5, 'NONE': 0, -1: 0}
ord_2_mapping = {'Freezing': 1, 'Cold': 2, 'Warm': 3, 'Hot': 4, 'Boiling Hot': 5, 'Lava Hot': 6, 'NONE': 0, -1: 0}

full_df['ord_1'] = full_df['ord_1'].map(ord_1_mapping)
full_df['ord_2'] = full_df['ord_2'].map(ord_2_mapping)

ord_3_col = ['ord_3', 'ord_4', 'ord_5']
for col in ord_3_col:
    # Use OrdinalEncoder after filling missing values
    full_df[col] = full_df[col].replace('NONE', chr(ord('a') - 1)).fillna(chr(ord('a') - 1))
    encoder = OrdinalEncoder(categories=[sorted(full_df[col].unique())], dtype=int)
    full_df[col] = encoder.fit_transform(full_df[[col]])


# 3.3: Frequency Encoding
print("--- 3.3: Frequency Encoding ---")
all_cat_cols = [col for col in full_df.columns if 'nom_' in col or 'bin_' in col or 'day' in col or 'month' in col]
for col in all_cat_cols:
    freq_map = full_df[col].value_counts().to_dict()
    full_df[f'{col}_freq'] = full_df[col].map(freq_map)


# 3.4: Cyclical Feature Engineering
print("--- 3.4: Cyclical Feature Engineering (day, month) ---")
full_df['day_sin'] = np.sin(2 * np.pi * full_df['day'] / 7.0)
full_df['day_cos'] = np.cos(2 * np.pi * full_df['day'] / 7.0)
full_df['month_sin'] = np.sin(2 * np.pi * full_df['month'] / 12.0)
full_df['month_cos'] = np.cos(2 * np.pi * full_df['month'] / 12.0)
full_df = full_df.drop(['day', 'month'], axis=1)



# 3.5: Feature Interactions
print("--- 3.5: Feature Interactions ---")
full_df['ord_1_nom_3'] = full_df['ord_1'].astype(str) + '_' + full_df['nom_3'].astype(str)
full_df['ord_1_ord_2'] = full_df['ord_1'].astype(str) + '_' + full_df['ord_2'].astype(str)



# 3.6: Converting Categorical Columns to 'category' (for CatBoost)
print("--- 3.6: Converting Categorical Columns to 'category' ---")
# Ensure bin_x columns are string type before converting to category
binary_cols_to_fix = [col for col in full_df.columns if 'bin_' in col]
for col in binary_cols_to_fix:
    full_df[col] = full_df[col].astype(str)

nominal_cols = [col for col in full_df.columns if 'nom_' in col or 'bin_' in col]
interaction_cols = ['ord_1_nom_3', 'ord_1_ord_2']
for col in nominal_cols + interaction_cols:
    full_df[col] = full_df[col].astype('category')


# Separate data back
X = full_df.iloc[:len(train_df)]
X_test = full_df.iloc[len(train_df):]


# Define Categorical Columns for CatBoost
categorical_features = [col for col in X.columns if X[col].dtype.name == 'category']
print(f"Total number of features: {len(X.columns)}")
print(f"Number of categorical features for CatBoost: {len(categorical_features)}")
print("--- Feature Engineering completed. ---")


## --- 4. Train ML Model (CatBoost Training) ---
print("\n--- 4. Train ML Model (Training with CatBoost) ---")
print("ğŸ”¥ Using StratifiedKFold and GPU training (requires GPU environment).")

# CatBoost parameters definition
cat_params = {
    'iterations': 3000,
    'learning_rate': 0.02,
    'depth': 6,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 0,
    'early_stopping_rounds': 1000,
    'task_type': 'GPU',     # Modification to enable GPU acceleration
    'metric_period': 20     # Reduce number of output metric prints to minimize slowdown
}

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds_cat = np.zeros(len(X))
test_preds_cat = np.zeros(len(X_test))


# Training and Cross-Validation loop
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
  
    print(f"\n--- FOLD {fold + 1}/{N_SPLITS} ---") 
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model_cat = CatBoostClassifier(**cat_params)
    
    model_cat.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  cat_features=categorical_features,
                  verbose=0
                  )
    
    val_preds = model_cat.predict_proba(X_val)[:, 1]
    oof_preds_cat[val_index] = val_preds
    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / N_SPLITS
    
    print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, val_preds):.6f}")
    
        # The preceding code block calculates val_preds and roc_auc_score
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds)}")
    
 
overall_auc_cat = roc_auc_score(y, oof_preds_cat)
print(f"\n===================================")
print(f"Overall OOF AUC (CatBoost): {overall_auc_cat:.6f}")
print(f"===================================")


# --- 5. Submission (Creating Submission File) ---
print("\n--- 5. Submission (Creating Submission File) ---")
submission_df_cat = pd.DataFrame({'id': test_ids, 'target': test_preds_cat})
submission_df_cat.to_csv('submission.csv', index=False)

print("âœ… 'submission.csv' file created successfully.")

