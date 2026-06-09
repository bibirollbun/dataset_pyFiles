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


import pandas as pd

# File paths
paths = {
    
    "Train": "/kaggle/input/mae-competencia-regresion-202513/Train real state.csv",
    "Test": "/kaggle/input/mae-competencia-regresion-202513/Test real state.csv"
}

# Loop through each file and display shape and memory usage
for name, path in paths.items():
    # Load the CSV file
    df = pd.read_csv(path)
    
    # Print shape: number of rows and columns
    print(f"{name} Shape: {df.shape}")
    
    # Print memory usage in MB
    memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"{name} Memory Usage: {memory:.2f} MB\n")


# Load train and test datasets
train = pd.read_csv("/kaggle/input/mae-competencia-regresion-202513/Train real state.csv")
test = pd.read_csv("/kaggle/input/mae-competencia-regresion-202513/Test real state.csv")

# 1. Check for missing (null) values in train and test
print("Missing values in Train:\n", train.isnull().sum().sort_values(ascending=False)[train.isnull().sum() > 0])
print("\nMissing values in Test:\n", test.isnull().sum().sort_values(ascending=False)[test.isnull().sum() > 0])

# 2. Check for duplicate rows in train and test
train_duplicates = train.duplicated().sum()
test_duplicates = test.duplicated().sum()

print(f"\nNumber of duplicate rows in Train: {train_duplicates}")
print(f"Number of duplicate rows in Test: {test_duplicates}")

# 3. Features in train but not in test
train_features = set(train.columns)
test_features = set(test.columns)

extra_train_features = train_features - test_features
print(f"\nFeatures present in Train but not in Test: {extra_train_features}")


train.info()


train.shape,test.shape


# Numeric features: dtype includes int, float
train_numeric = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
test_numeric = test.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Categorical features: dtype usually object or category
train_categorical = train.select_dtypes(include=['object', 'category']).columns.tolist()
test_categorical = test.select_dtypes(include=['object', 'category']).columns.tolist()

print("Train Numeric Features:", train_numeric)
print("Train Categorical Features:", train_categorical)

print("\nTest Numeric Features:", test_numeric)
print("Test Categorical Features:", test_categorical)


feature_names = []
dtypes = []
null_counts = []
unique_values = []

for col in train.columns:
    feature_names.append(col)
    dtypes.append(train[col].dtype)
    null_counts.append(train[col].isnull().sum())
    
    unique_vals = train[col].dropna().unique()
    # Take only first 10 unique values as list and convert to string
    unique_sample = unique_vals[:10]
    unique_values.append(", ".join(map(str, unique_sample)))

# Create summary DataFrame
summary_df = pd.DataFrame({
    "Feature Name": feature_names,
    "Data Type": dtypes,
    "Null Count": null_counts,
    "Unique Values (max 10)": unique_values
})

summary_df


train.describe().T


# Loop through each column and show unique values (max 10)
print("ğŸ”� Unique values for each train feature (Max 10 values shown):\n")

for col in train.columns:
    unique_vals = train[col].dropna().unique()
    n_unique = len(unique_vals)
    dtype = train[col].dtype

    print(f"\nğŸ“Œ Feature: {col}")
    print(f" - Data Type: {dtype}")
    print(f" - Unique Values ({n_unique} total): ", end="")

    if n_unique > 10:
        print(f"{unique_vals[:10].tolist()} ... [+{n_unique - 10} more]")
    else:
        print(unique_vals.tolist())



import warnings
warnings.filterwarnings("ignore")

feature_names = []
dtypes = []
null_counts = []
unique_values = []
unique_counts = []
most_freq_values = []
freq_most_freq = []

for col in train.columns:
    feature_names.append(col)
    dtypes.append(train[col].dtype)
    null_counts.append(train[col].isnull().sum())
    
    unique_vals = train[col].dropna().unique()
    unique_values.append(", ".join(map(str, unique_vals[:10])))
    
    unique_counts.append(train[col].nunique())
    
    mode_val = train[col].mode()
    if not mode_val.empty:
        most_freq_values.append(mode_val.iloc[0])
        freq_most_freq.append(train[col].value_counts().iloc[0])
    else:
        most_freq_values.append(None)
        freq_most_freq.append(0)

summary_df = pd.DataFrame({
    "Feature Name": feature_names,
    "Data Type": dtypes,
    "Null Count": null_counts,
    "Null Percentage": (pd.Series(null_counts) / len(train)) * 100,
    "Unique Count": unique_counts,
    "Unique Values (max 10)": unique_values,
    "Most Frequent Value": most_freq_values,
    "Freq of Most Frequent": freq_most_freq,
})

# Add numeric summary stats
numeric_stats = train.describe().T
summary_df = summary_df.merge(numeric_stats[['mean', '50%', 'std']], left_on='Feature Name', right_index=True, how='left')
summary_df.rename(columns={'50%': 'Median'}, inplace=True)

# Add is constant flag
summary_df['Is Constant'] = summary_df['Unique Count'] == 1

# Feature type category
def feature_type(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return 'Numeric'
    elif pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
        return 'Categorical'
    else:
        return 'Other'

summary_df['Feature Type'] = summary_df['Data Type'].apply(feature_type)

summary_df


numeric_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col != 'id']

total_rows = train.shape[0]

for col in numeric_features:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
    count = outliers.shape[0]
    percentage = (count / total_rows) * 100

    print(f"{col}: {count} outliers, {percentage:.2f}%")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import math
import warnings
warnings.filterwarnings("ignore")

# Separate features
numeric_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col != 'Unnamed: 0']
categorical_features = train.select_dtypes(include=['object', 'category']).columns.tolist()

# Combine features and mark types
all_features = [(col, 'numeric') for col in numeric_features] + [(col, 'categorical') for col in categorical_features]

# Plot settings
features_per_page = 8
total_pages = math.ceil(len(all_features) / features_per_page)

for page in range(total_pages):
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    start = page * features_per_page
    end = start + features_per_page
    selected_features = all_features[start:end]

    for ax, (col, ftype) in zip(axes, selected_features):
        if ftype == 'numeric':
            data = train[col].dropna()
            skw = skew(data)
            krt = kurtosis(data)
            sns.histplot(data, bins=30, kde=True, ax=ax, color='skyblue')
            ax.set_title(f"{col}\nSkewness: {skw:.2f}, Kurtosis: {krt:.2f}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        elif ftype == 'categorical':
            counts = train[col].value_counts()
            percentages = counts / counts.sum() * 100
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette='Set2')
            for idx, val in enumerate(counts.values):
                pct = percentages.values[idx]
                ax.text(idx, val + max(counts.values)*0.01, f"{pct:.1f}%", ha='center', fontsize=9)
            ax.set_title(f"{col} (n={counts.sum()})")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

    # Remove any unused subplots
    for i in range(len(selected_features), 8):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle(f"Univariate Analysis (Page {page+1}/{total_pages})", fontsize=16, y=1.02)
    plt.show()


# Define target
target = 'SalePrice'

# Separate feature types
numeric_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col not in ['Unnamed: 0', target]]
categorical_features = [col for col in train.select_dtypes(include=['object', 'category']).columns if col != 'Unnamed: 0']

# Combine feature list with types
all_features = [(col, 'numeric') for col in numeric_features] + [(col, 'categorical') for col in categorical_features]

# Plot parameters
features_per_page = 8
total_pages = math.ceil(len(all_features) / features_per_page)

for page in range(total_pages):
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes = axes.flatten()
    start = page * features_per_page
    end = min(start + features_per_page, len(all_features))
    selected_features = all_features[start:end]

    for ax, (col, ftype) in zip(axes, selected_features):
        if ftype == 'numeric':
            # Boxplot for numeric feature vs target
            sns.boxplot(data=train, x=col, y=target, ax=ax, palette='Set2')
            ax.set_title(f"{col} vs {target}", fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel(target)
        elif ftype == 'categorical':
            # Boxplot for categorical feature vs target
            sns.boxplot(data=train, x=col, y=target, ax=ax, palette='Set2')
            ax.set_title(f"{col} by {target}", fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel(target)

    # Remove any unused subplot axes
    for i in range(len(selected_features), 8):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle(f"Bivariate Analysis: {target} vs Features (Page {page+1}/{total_pages})", fontsize=16, y=1.02)
    plt.show()



import math
import matplotlib.pyplot as plt
import seaborn as sns

target = 'SalePrice'

# Identify numeric features (excluding 'id' and target itself)
numeric_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns if col not in ['Unnamed: 0', target]]

# Plot config
features_per_page = 24
total_pages = math.ceil(len(numeric_features) / features_per_page)

for page in range(total_pages):
    fig, axes = plt.subplots(6, 4, figsize=(18, 20))
    axes = axes.flatten()
    start = page * features_per_page
    end = min(start + features_per_page, len(numeric_features))
    selected_features = numeric_features[start:end]

    for ax, col in zip(axes, selected_features):
        sns.scatterplot(data=train, x=col, y=target, ax=ax, color='steelblue')
        ax.set_title(f"{col} vs {target}", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel(target)

    for i in range(len(selected_features), 8):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle(f"Scatter Plot - Numeric Features vs {target} (Page {page+1}/{total_pages})", fontsize=16, y=1.02)
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

selected_numeric = numeric_features[:5]  

plt.figure(figsize=(14, 12))

plt.figure(figsize=(14, 12))

for i, feature in enumerate(selected_numeric, 1):
    plt.subplot(3, 2, i)
    sns.kdeplot(data=train, x=feature, fill=True, alpha=0.4, color='skyblue')
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()



# Function to calculate approximate number of days until sold
def compute_days_until_sold(df):
    return (df['YrSold'] - df['YearBuilt']) * 365 + (df['MonthSold'] * 30)

# Apply to both train and test datasets
train['DaysUntilSold'] = compute_days_until_sold(train)
test['DaysUntilSold'] = compute_days_until_sold(test)


import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Avoid log(0) by adding 1 (if zero days exist)
train['DaysUntilSold_log'] = np.log1p(train['DaysUntilSold'])

# Scale the log-transformed feature
scaler = StandardScaler()
train['DaysUntilSold_log_scaled'] = scaler.fit_transform(train[['DaysUntilSold_log']])

# Plot histograms before and after transformation
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(train['DaysUntilSold'], bins=50, kde=True)
plt.title("Original DaysUntilSold Distribution")

plt.subplot(1, 2, 2)
sns.histplot(train['DaysUntilSold_log'], bins=50, kde=True, color='orange')
plt.title("Log-Transformed DaysUntilSold Distribution")

plt.tight_layout()
plt.show()
from scipy.stats import pearsonr

plt.figure(figsize=(10, 6))
sns.scatterplot(data=train, x='DaysUntilSold_log', y='SalePrice', alpha=0.6)
sns.regplot(data=train, x='DaysUntilSold_log', y='SalePrice', scatter=False, color='red')
plt.title("DaysUntilSold (Log-transformed) vs SalePrice")
plt.xlabel("Log(DaysUntilSold + 1)")
plt.ylabel("SalePrice")
plt.show()

# Correlation
corr_log, pval_log = pearsonr(train['DaysUntilSold_log'], train['SalePrice'])
print(f"Pearson correlation after log transform: {corr_log:.4f} (p-value: {pval_log:.4e})")



# Step 2: Log transform
test['DaysUntilSold_log'] = np.log1p(test['DaysUntilSold'])

# Step 3: Use SAME scaler trained on train data
test['DaysUntilSold_log_scaled'] = scaler.transform(test[['DaysUntilSold_log']])


# ================================
# Imports and Global Config
# ================================
import pandas as pd
import numpy as np
import random
import os

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Seed Everything for Reproducibility
# ================================
SEED = 42
def seed_everything(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()

# ================================
# Configuration
# ================================
TARGET = 'SalePrice'
ID_COL = 'Unnamed: 0'
N_FOLDS = 10
VERBOSE = True
OUTPUT_FILE = 'submission_stack.csv'

# ================================
# Data Preparation
# ================================
X = train.drop(columns=[TARGET])
y = train[TARGET]
X_test = test.copy()

cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# One-hot encoding
encoder = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_features)
], remainder='passthrough')

X_encoded = encoder.fit_transform(X)
X_test_encoded = encoder.transform(X_test.drop(columns=[TARGET], errors='ignore'))

# ================================
# Define Base Models
# ================================
base_models = [
    ('lgbm', LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED
    )),
    ('xgb', XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        random_state=SEED,
        verbosity=0
    )),
    ('cat', CatBoostRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5,
        random_seed=SEED,
        verbose=0
    ))
]

# ================================
# Cross-validation and Stacking
# ================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

meta_features = np.zeros((len(X), len(base_models)))
test_meta_features = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models):
    print(f"\nğŸ”� Training base model: {name.upper()}")
    fold_preds = np.zeros(len(X))
    test_preds_fold = np.zeros((len(X_test), N_FOLDS))
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded)):
        X_train_fold, X_val_fold = X_encoded[train_idx], X_encoded[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        if name == 'cat':
            model.fit(X_train_fold, y_train_fold,
                      eval_set=(X_val_fold, y_val_fold),
                      use_best_model=False, verbose=0)
        else:
            model.fit(X_train_fold, y_train_fold)

        val_pred = model.predict(X_val_fold)
        test_pred = model.predict(X_test_encoded)

        fold_preds[val_idx] = val_pred
        test_preds_fold[:, fold] = test_pred

        mae = mean_absolute_error(y_val_fold, val_pred)
        fold_maes.append(mae)

        if VERBOSE:
            print(f"   Fold {fold+1}/{N_FOLDS} MAE: {mae:.4f}")

    meta_features[:, i] = fold_preds
    test_meta_features[:, i] = test_preds_fold.mean(axis=1)

    print(f"âœ… {name.upper()} Mean MAE: {np.mean(fold_maes):.4f} Â± {np.std(fold_maes):.4f}")

# ================================
# Train Meta-Model
# ================================
print("\nğŸ§  Training meta-model: Ridge")
meta_model = Ridge(alpha=10.0, random_state=SEED)
meta_model.fit(meta_features, y)

meta_oof = meta_model.predict(meta_features)
final_test_preds = meta_model.predict(test_meta_features)

meta_mae = mean_absolute_error(y, meta_oof)
print(f"\nâœ… Final Stacking MAE (on full OOF): {meta_mae:.4f}")

# ================================
# Submission
# ================================
submission = pd.DataFrame({
    'Id': test[ID_COL],
    'Predicted': final_test_preds
})
#submission.to_csv(OUTPUT_FILE, index=False)
#print(f"\nğŸ“� Submission saved as '{OUTPUT_FILE}'")





