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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold


from xgboost import XGBRegressor

from ydf import RandomForestLearner, GradientBoostedTreesLearner
import ydf

# Display settings and warnings configuration
pd.set_option('display.max_columns', 100)
plt.style.use('ggplot')
warnings.filterwarnings('ignore')



train = pd.read_csv('../input/playground-series-s5e2/train.csv', index_col=0)
test = pd.read_csv('../input/playground-series-s5e2/test.csv', index_col=0)

print("Train dataset shape:", train.shape)
display(train.head())
print("Missing values in train:\n", train.isnull().sum())
print("****************************************************************************************************************************************")
print("*****************************************************************************************************************************************")
print("****************************************************************************************************************************************")
print("Test dataset shape:", test.shape)
display(test.head())
print("Missing values in test:\n", test.isnull().sum())



cat_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

for col in cat_cols:
    train[col] = train[col].fillna('Unknown').astype('category')
    test[col] = test[col].fillna('Unknown').astype('category')


train['Capacity_per_Compartment'] = train['Weight Capacity (kg)'] / (train['Compartments'] + 1e-6)
test['Capacity_per_Compartment'] = test['Weight Capacity (kg)'] / (test['Compartments'] + 1e-6)

train['Brand_Length'] = train['Brand'].astype(str).apply(len)
test['Brand_Length'] = test['Brand'].astype(str).apply(len)


gc.collect()  



# 4.1 Distribution of the Target Variable (Price)
# --------------------------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(train['Price'], kde=True, bins=50, color='skyblue')
plt.title("Distribution of Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.axvline(x=150, color='red', linestyle='--', label='Price Censoring at 150')
plt.legend()
plt.show()

# 4.2 Missing Value Visualization (Train Dataset)
# --------------------------------------------------
plt.figure(figsize=(12, 4))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in Train Dataset")
plt.show()

# 4.3 Correlation Analysis among Numerical Features
# --------------------------------------------------
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(8, 6))
sns.heatmap(train[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for Numerical Features")
plt.show()

# 4.4 Boxplots: Price vs. Some Categorical Features
# --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(cat_cols[:6]):  # using the first 6 categorical features
    sns.boxplot(x=col, y='Price', data=train, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f"Price vs {col}")
    axes[i//3, i%3].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# 4.5 Additional Exploration for 'Color'
# --------------------------------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(x="Color", y="Price", data=train)
plt.xticks(rotation=45)
plt.title("Price vs Color")
plt.show()



%%time
# ----- Model 1: RandomForestLearner -----
skf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
ydf.verbose(-1)  # Suppress verbose output

rf_scores = []
rf_test_preds = []
print("Starting RandomForestLearner cross-validation...\n")
for fold, (train_idx, valid_idx) in enumerate(skf.split(train)):
    print(f"------------ RF: Fold {fold} ------------")
    X_train_rf = train.iloc[train_idx].copy()
    X_valid_rf = train.iloc[valid_idx].copy()
    
    # Correctly assign the trained model returned by train()
    model_rf = RandomForestLearner(
        label='Price', 
        task=ydf.Task.REGRESSION, 
        num_threads=10, 
        num_trees=1000
    ).train(X_train_rf)
    preds_valid_rf = model_rf.predict(X_valid_rf)
    score = mean_squared_error(X_valid_rf['Price'], preds_valid_rf, squared=False)
    print('Fold:', fold, 'RMSE:', score)
    rf_scores.append(score)
    # Save test predictions for this fold
    rf_test_preds.append(model_rf.predict(test))

rf_avg_rmse = np.mean(rf_scores)
rf_std_rmse = np.std(rf_scores)
print(f"\nRF 5-fold Average RMSE: {rf_avg_rmse:.4f}, Std: {rf_std_rmse:.4f}")


# Create final submission for RandomForestLearner
submission_rf = pd.read_csv('../input/playground-series-s5e2/sample_submission.csv')
submission_rf["Price"] = np.mean(rf_test_preds, axis=0)
submission_rf.to_csv("baseline_RF_sub.csv", index=False)
print("\nRandomForestLearner Submission Preview:")
display(submission_rf.head())



# ----- Model 2: GradientBoostedTreesLearner -----
ydf.verbose(-1)  # Again, suppress verbose output

gb_scores = []
gb_test_preds = []
print("Starting GradientBoostedTreesLearner cross-validation...\n")

for fold, (train_idx, valid_idx) in enumerate(skf.split(train)):
    print(f"------------ GB: Fold {fold} ------------")
    X_train_gb = train.iloc[train_idx].copy()
    X_valid_gb = train.iloc[valid_idx].copy()
    
    # Chain the train() method to get the trained model
    model_gb = GradientBoostedTreesLearner(
        label='Price', 
        task=ydf.Task.REGRESSION, 
        num_threads=10, 
        num_trees=1000
    ).train(X_train_gb)
    
    # Use the trained model for prediction
    preds_valid_gb = model_gb.predict(X_valid_gb)
    rmse_gb = mean_squared_error(X_valid_gb['Price'], preds_valid_gb, squared=False)
    print(f"Fold {fold} RMSE: {rmse_gb:.4f}")
    gb_scores.append(rmse_gb)
    
    # Save test predictions for this fold
    gb_test_preds.append(model_gb.predict(test))
    
gb_avg_rmse = np.mean(gb_scores)
gb_std_rmse = np.std(gb_scores)
print(f"\nGB 5-fold Average RMSE: {gb_avg_rmse:.4f}, Std: {gb_std_rmse:.4f}")



# Create final submission for GradientBoostedTreesLearner
submission_gb = pd.read_csv('../input/playground-series-s5e2/sample_submission.csv')
submission_gb["Price"] = np.mean(gb_test_preds, axis=0)
submission_gb.to_csv("baseline_GB_sub.csv", index=False)
print("\nGradientBoostedTreesLearner Submission Preview:")
display(submission_gb.head())



# ----- Model 3: XGBRegressor -----
# Prepare features and target
X = train.drop(columns=["Price"], axis=1).copy()
y = train["Price"].copy()

# Define hyperparameters (as originally tuned)
xgb_params = {
    'objective': 'reg:absoluteerror',
    'n_estimators': 675,
    'max_depth': 12,
    'learning_rate': 0.0647368285818005,
    'gamma': 5.581559809586505,
    'min_child_weight': 31,
    'colsample_bytree': 0.467360303051405,
    'n_jobs': -1,
    'enable_categorical': True
}

xgb_scores = []
xgb_test_preds = []

# We use 10-fold CV here
skf_xgb = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
fold_number = 0
print("Starting XGBRegressor cross-validation...\n")
for train_idx, valid_idx in skf_xgb.split(X, y):
    fold_number += 1
    X_train_xgb = X.iloc[train_idx].copy()
    X_valid_xgb = X.iloc[valid_idx].copy()
    y_train_xgb = y.iloc[train_idx]
    y_valid_xgb = y.iloc[valid_idx]
    
    model_xgb = XGBRegressor(**xgb_params)
    model_xgb.fit(X_train_xgb, y_train_xgb)
    
    preds_valid_xgb = model_xgb.predict(X_valid_xgb)
    rmse_xgb = mean_squared_error(y_valid_xgb, preds_valid_xgb, squared=False)
    print(f"Fold {fold_number} RMSE: {rmse_xgb:.4f}")
    xgb_scores.append(rmse_xgb)
    
    # Save test predictions for this fold
    xgb_test_preds.append(model_xgb.predict(test))
    
xgb_avg_rmse = np.mean(xgb_scores)
xgb_std_rmse = np.std(xgb_scores)
print(f"\nXGBRegressor 10-fold Average RMSE: {xgb_avg_rmse:.4f}, Std: {xgb_std_rmse:.4f}")



# Create final submission for XGBRegressor
submission_xgb = pd.read_csv('../input/playground-series-s5e2/sample_submission.csv')
submission_xgb["Price"] = np.mean(xgb_test_preds, axis=0)
submission_xgb.to_csv("baseline_xgb_sub.csv", index=False)
print("\nXGBRegressor Submission Preview:")
display(submission_xgb.head())




