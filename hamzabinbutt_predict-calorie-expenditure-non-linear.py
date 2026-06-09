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


# Standard Libraries
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Scikit-learn Modules
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_squared_log_error,
    mean_squared_error,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.cluster import KMeans

# Gradient Boosting Libraries
from catboost import CatBoostRegressor, Pool

# Plotly for Interactive Visualization
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


# Suppress Warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# File paths
train_path = "/kaggle/input/playground-series-s5e5/train.csv"
test_path = "/kaggle/input/playground-series-s5e5/test.csv"

# Read CSV files
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Drop 'id' column from both datasets
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

# Display the first few rows
print("Train DataFrame:")
print(train_df.head())

print("\nTest DataFrame:")
print(test_df.head())


# Compute correlation with Calories
correlation = train_df.corr(numeric_only=True)['Calories'].sort_values(ascending=False)

print("Correlation of each variable with Calories:")
print(correlation)



# Compute correlation matrix (numeric only)
corr = train_df.corr(numeric_only=True)

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title("Half-Matrix Correlation Heatmap (train_df)", fontsize=14)
plt.tight_layout()
plt.show()


# Check missing values (NaN) per column
nan_counts = train_df.isna().sum()

# Check infinite values per column
inf_counts = train_df.isin([np.inf, -np.inf]).sum()

# Combine into one report
missing_report = nan_counts.to_frame(name='NaN_Count')
missing_report['Inf_Count'] = inf_counts

print("Missing and Infinite Values per column:")
print(missing_report)

# Summary: total missing (NaN + Inf)
missing_report['Total_Missing'] = missing_report['NaN_Count'] + missing_report['Inf_Count']
print("\nTotal missing values (NaN + Inf) per column:")
print(missing_report[['Total_Missing']])



# Select numeric columns only
numeric_cols = train_df.select_dtypes(include=['number']).columns

# Plot histograms
train_df[numeric_cols].hist(bins=15, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numeric Variables', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


train_df.describe()


# --- 1. Clustering for multimodal variables ---
multimodal_vars = ['Height', 'Weight', 'Duration', 'Heart_Rate']

# Clustering for multimodal variables with explicit n_init
for var in multimodal_vars:
    X = train_df[[var]].dropna().values
    kmeans = KMeans(n_clusters=15, random_state=42, n_init='auto')  # Explicit n_init to suppress warning
    clusters = kmeans.fit_predict(X)
    train_df.loc[train_df[var].notna(), f'{var}_cluster'] = clusters


# --- 2. Right-skewed transformation for Calories ---
# Log transform Calories (add 1 to avoid log(0))
train_df['Calories_log'] = np.log1p(train_df['Calories'])
# --- 3. Left-skewed transformation for Body_Temp ---
# Reflect data to handle left skewness and then apply log transform
max_temp = train_df['Body_Temp'].max()
train_df['Body_Temp_reflect_log'] = np.log1p(max_temp + 1 - train_df['Body_Temp'])

pt = PowerTransformer(method='yeo-johnson')

# Apply PowerTransformer (Yeo-Johnson) for Body_Temp
train_df['Body_Temp_yeojohnson'] = pt.fit_transform(train_df[['Body_Temp']])

# --- Optional: Binning example for one multimodal variable (Height) ---
# Quantile binning into 3 bins
train_df['Height_bin'] = pd.qcut(train_df['Height'], q=15, labels=False)

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply label encoding to 'Sex' column
train_df['Sex'] = le.fit_transform(train_df['Sex'])

train_df.drop('Calories', axis=1, inplace=True)

# Show resulting dataframe columns to verify
train_df.head()



# List of new categorical/clustered variables to plot bar charts for
bar_vars = [
    'Height_cluster',
    'Weight_cluster',
    'Duration_cluster',
    'Heart_Rate_cluster',
    'Height_bin'
]

# Plot bar charts for cluster and bin variables (categorical)
plt.figure(figsize=(15, 8))
for i, var in enumerate(bar_vars, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=var, data=train_df, palette='Set2')
    plt.title(f'Bar Chart of {var}')
    plt.xlabel(var)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


# For transformed continuous variables, plot histograms to show distribution
cont_vars = [
    'Calories_log',
    'Body_Temp_reflect_log',
    'Body_Temp_yeojohnson'
]

plt.figure(figsize=(15, 8))
for i, var in enumerate(cont_vars, 1):
    plt.subplot(2, 2, i)
    sns.histplot(train_df[var].dropna(), kde=True, color='skyblue')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
plt.tight_layout()
plt.show()



y = train_df['Calories_log']  
# Drop the target column from the feature set (X)
X = train_df.drop('Calories_log', axis=1)
# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Optional: Check the shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Ensure all inf/-inf are removed
X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaNs in X or y
X = X.dropna()
y = y.loc[X.index]

# Initialize K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Store RMSEs
rmse_list = []

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Define CatBoost Pool
    train_pool = Pool(X_train_fold, y_train_fold)
    val_pool = Pool(X_val_fold, y_val_fold)

    # Initialize and train model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=0,
        random_state=42
    )
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    # Predict and evaluate
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    rmse_list.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

# Final average RMSE
print(f"\nAverage RMSE across folds: {np.mean(rmse_list):.4f}")



# --- 1. Clustering for multimodal variables ---
multimodal_vars = ['Height', 'Weight', 'Duration', 'Heart_Rate']

for var in multimodal_vars:
    X_test_var = test_df[[var]].dropna().values
    kmeans = KMeans(n_clusters=15, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_test_var)
    test_df.loc[test_df[var].notna(), f'{var}_cluster'] = clusters

# --- 2. Right-skewed transformation for Calories ---
#test_df['Calories_log'] = np.log1p(test_df['Calories'])

# --- 3. Left-skewed transformation for Body_Temp ---
max_temp = train_df['Body_Temp'].max()  # use max from train set
test_df['Body_Temp_reflect_log'] = np.log1p(max_temp + 1 - test_df['Body_Temp'])

# --- 4. Apply PowerTransformer (Yeo-Johnson) using the same transformer from train ---
pt = PowerTransformer(method='yeo-johnson')  # ensure you fit on train first
pt.fit(train_df[['Body_Temp']])
test_df['Body_Temp_yeojohnson'] = pt.transform(test_df[['Body_Temp']])

# --- 5. Binning for multimodal variable (Height) ---
# Use same bin edges as train
bin_edges = np.quantile(train_df['Height'].dropna(), np.linspace(0, 1, 16))  # 15 bins = 16 edges
test_df['Height_bin'] = pd.cut(test_df['Height'], bins=bin_edges, labels=False, include_lowest=True)

# --- 6. Label encoding for 'Sex' using same encoder ---
test_df['Sex'] = le.transform(test_df['Sex'])

# --- 7. Drop original Calories column ---
#test_df.drop('Calories', axis=1, inplace=True)

# Preview
test_df.head()


print(test_df.shape)


# Predict on test set (log-transformed scale)
y_pred_test = model.predict(test_df)

# Inverse the log1p transformation to get predictions in original scale
y_pred_test_original = np.expm1(y_pred_test)


# Print first 10 predictions in both scales
print("Log-scale predictions (first 10):")
print(y_pred_test[:10])

print("\nOriginal scale predictions (first 10):")
print(y_pred_test_original[:10])


# Load sample submission file
sample_submission_path = '/kaggle/input/playground-series-s5e5/sample_submission.csv'
submission_df = pd.read_csv(sample_submission_path)

# Replace second column (usually target) with predictions
# Assuming second column is at index 1
submission_df.iloc[:, 1] = y_pred_test_original

# Save new submission file to working directory
submission_df.to_csv('/kaggle/working/cat-(0.0596cv).csv', index=False)

