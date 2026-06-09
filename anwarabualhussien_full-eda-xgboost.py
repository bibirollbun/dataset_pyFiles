from IPython.display import display, HTML

display(HTML("""
<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/ABUALHUSSEIN/predicting-road-accident-risk-kaggle/refs/heads/main/road.png" width="1000">
</div>
"""))



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

# Settings
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)



train  = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")  


# Quick look
train.head()


print("Shape of dataset:", train.shape)


train.info()


# Let's identify the categorical columns you mentioned
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']


print("--- Checking Levels of Categorical Features ---")
for col in categorical_features:
    num_levels = train[col].nunique()  # Get the number of unique levels
    levels = train[col].unique()      # Get the actual unique levels
    
    print(f"\nFeature: '{col}'")
    print(f"  Number of unique levels: {num_levels}")
    print(f"  Levels: {levels}")
    print("-" * 30)


train.describe()


train.info()


# Missing values
print("Missing Values:", train.isnull().sum())


#Duplicate rows
print("Duplicate rows:", train.duplicated().sum())



target = "accident_risk"

def plot_target_distribution(train, target, bins=50):
    """
    Generates a comprehensive plot for analyzing a regression target variable.
    
    The plot includes a histogram, KDE, boxplot, and key statistical annotations.
    
    Parameters:
    - train (pd.DataFrame): The input dataframe.
    - target (str): The name of the target column.
    - bins (int): The number of bins for the histogram.
    """
    # --- Calculate Statistics ---
    mean_val = train[target].mean()
    
    median_val = train[target].median()
    std_val = train[target].std()
    skew_val = train[target].skew()
    kurt_val = train[target].kurt()

    # --- Create the plot ---
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, 
        gridspec_kw={'height_ratios': (0.8, 0.2)}
    )
    
    # --- Histogram and KDE (Top Plot) ---
    sns.histplot(train[target], ax=ax_hist, kde=True, bins=bins, line_kws={'linewidth': 2})
    
    # Add vertical lines for mean and median
    ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax_hist.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
    
    ax_hist.set_title(f'Distribution of {target}', fontsize=16, weight='bold')
    ax_hist.set_ylabel('Frequency', fontsize=12)
    ax_hist.legend(loc='upper right')
    ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
    ax_hist.set_xlabel('')  # Hide x-label for the top plot

    # --- Statistical Annotations ---
    stats_text = (
        f"Std. Dev: {std_val:.2f}\n"
        f"Skewness: {skew_val:.2f}\n"
        f"Kurtosis: {kurt_val:.2f}"
    )
    ax_hist.text(0.97, 0.97, stats_text, transform=ax_hist.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

    # --- Boxplot (Bottom Plot) ---
    sns.boxplot(x=train[target], ax=ax_box, color='skyblue')
    ax_box.set_xlabel(target, fontsize=12)
    ax_box.set_ylabel(' ', fontsize=12)

    # --- Final Touches ---
    plt.suptitle(f'Detailed Analysis of Target Variable: {target}', fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()
plot_target_distribution(train, target)



# ========================
# 4. Feature Distributions
# ========================


# Select numeric features (excluding ID if present)
num_features = train.select_dtypes(include=[np.number]).columns.tolist()
# Columns to exclude
exclude_cols = ["id","accident_risk"]
num_features = [col for col in num_features if col not in exclude_cols]

# Define grid size automatically (rows & cols)

n_features = len(num_features)
n_cols = 2
n_rows = int(np.ceil(n_features / n_cols))

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

# Plot each feature
for i, col in enumerate(num_features):
    sns.histplot(train[col], bins=30, kde=True, ax=axes[i], color="skyblue")
    axes[i].set_title(f"Distribution of {col}", fontsize=12, weight="bold")
    axes[i].set_xlabel("")
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)

# Remove empty subplots (if any)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Feature Distributions", fontsize=16, weight="bold", y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# Select numeric columns, excluding 'id' if present

num_features = train.select_dtypes(include=[np.number]).columns.tolist()
# Columns to exclude
exclude_cols = ["id","accident_risk"]
numerical_cols  = [col for col in num_features if col not in exclude_cols]

# Define color palette
palette = sns.color_palette("husl", len(numerical_cols))

# Grid layout: e.g., 4 columns
ncols = 2
nrows = -(-len(numerical_cols) // ncols)  # ceiling division

plt.figure(figsize=(5*ncols, 4*nrows))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(nrows, ncols, i)
    sns.boxplot(y=train[col], color=palette[i-1], showfliers=True, whis=1.5)
    plt.title(f'Boxplot of {col}', fontsize=12)
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.suptitle("Boxplots of Numerical Features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# ====================================================
# 6. Correlation with Target Variable(Visualized)
# ====================================================

# Assuming 'target_col' is the name of your target variable string
target_col = 'accident_risk' # Replace with your actual target name

# We can reuse the corr_matrix from before
if 'corr_matrix' not in locals():
    corr_matrix = train[num_features].corr()

# Get correlations with the target, drop the target's self-correlation, and sort
target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
# You can still print the sorted values for exact numbers
print(f"\n--- Correlation with {target_col} ---")
print(target_corr)


num_features = train.select_dtypes(include=[np.number]).columns.tolist()

# Columns to exclude

exclude_cols = ["id"]
num_features = [col for col in num_features if col not in exclude_cols]

# ==================================
# 5. Enhanced Correlation Analysis
# ==================================


# --- You already have this from the previous step ---
# Assuming 'num_features' is your list of numeric columns
# ---------------------------------------------------

# 1. Calculate the correlation matrix once
corr_matrix = train[num_features].corr()

# 2. Create a mask to hide the redundant upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 3. Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# 4. **Expert Tip**: Choose a better diverging colormap and center it
# 'vlag' is a great blue-white-red palette. 'icefire' is another good one.
# Centering at 0 ensures that 0 correlation is neutral (white).
sns.heatmap(corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap="vlag", # A better color palette for correlations
            vmin=-1, vmax=1, # Lock the color scale
            center=0,
            linewidths=.5, # Add lines between cells
            cbar_kws={"shrink": .8}) # Shrink the color bar a bit

plt.title("Correlation Heatmap of Numeric Features", fontsize=16, weight="bold")
plt.xticks(rotation=45, ha='right') # Rotate labels for readability
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# =============================================================================
# 1. SETUP AND IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import xgboost as xgb  # <-- CHANGED IMPORT
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# For reproducibility of results
SEED = 42

print("Libraries imported successfully.")


# Store test IDs for final submission file
test_ids = test['id']


# =============================================================================
# 3. PREPROCESSING PIPELINE (This section is identical)
# =============================================================================
print("\nStarting preprocessing...")


# This is safer and can be run multiple times
train_df = train.drop("id", axis=1, errors='ignore')
test_df = test.drop("id", axis=1, errors='ignore')

# Separate features (X) and target (y)
X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']
X_test = test_df

# --- Identify feature types ---
categorical_features = X.select_dtypes(include=['object', 'bool']).columns
numerical_features = X.select_dtypes(include=np.number).columns
print(f"Categorical Features: {list(categorical_features)}")
print(f"Numerical Features: {list(numerical_features)}")

# --- One-Hot Encode Categorical Features ---
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# --- Align columns between train and test data ---
train_cols = X.columns
test_cols = X_test.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test: X_test[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train: X[c] = 0
X_test = X_test[train_cols]

# --- Scale Numerical Features ---
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print("\nPreprocessing complete.")
print("Shape of final training features (X):", X.shape)
print("Shape of final test features (X_test):", X_test.shape)


# =============================================================================
# 4. MODEL TRAINING (XGBOOST WITH 5-FOLD CV AND GPU)
# =============================================================================
print("\nStarting model training with XGBoost 5-Fold Cross-Validation...")

NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_test.shape[0])
feature_importances = pd.DataFrame(index=X.columns)

# XGBoost parameters - configured for GPU
xgb_params = {
    'objective': 'reg:squarederror', # Objective for regression
    'eval_metric': 'rmse',           # Evaluation metric
    'n_estimators': 10000,           # High number, will be stopped by early stopping
    'learning_rate': 0.01,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'lambda': 1,
    'alpha': 1,
    'seed': SEED,
    'n_jobs': -1,
    'tree_method': 'gpu_hist',       # <-- KEY PARAMETER FOR GPU USAGE
    'predictor': 'gpu_predictor'     # Also helps ensure predictions run on GPU
}


for n_fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = xgb.XGBRegressor(**xgb_params) # <-- Use XGBRegressor
    
    # The fit call is slightly different for XGBoost
    model.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)],
              early_stopping_rounds=200, # Early stopping parameter
              verbose=False)             # Suppress training output

    # Store predictions
    oof_preds[valid_idx] = model.predict(X_valid)
    sub_preds += model.predict(X_test) / NFOLDS
    
    # Store feature importances
    feature_importances[f'fold_{n_fold+1}'] = model.feature_importances_
    
    fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_idx]))
    print(f"Fold {n_fold+1} RMSE: {fold_rmse}")

# Calculate overall Out-of-Fold RMSE
overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOverall Out-of-Fold CV RMSE with XGBoost: {overall_rmse}")



# =============================================================================
# 5. FEATURE IMPORTANCE VISUALIZATION
# =============================================================================
print("\nVisualizing feature importances...")

feature_importances['mean'] = feature_importances.mean(axis=1)
feature_importances.sort_values('mean', ascending=False, inplace=True)

plt.figure(figsize=(12, 16))
sns.barplot(x='mean', y=feature_importances.index, data=feature_importances)
plt.title('XGBoost Feature Importance (Mean over 5 Folds)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True)
plt.show()


print("\nGenerating submission file...")

submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': sub_preds})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("Top 5 rows of the submission file:")
print(submission_df.head())

