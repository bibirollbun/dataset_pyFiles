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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings

# Configuration
warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid", palette="muted")
pd.set_option('display.max_columns', None)




# Load Data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Drop ID column from Train (keep it in Test for submission later)
train_id = train_df['id']
test_id = test_df['id']

train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

# Target Balance Check
target_counts = train_df['diagnosed_diabetes'].value_counts(normalize=True)
print("\nTarget Distribution:")
print(target_counts)


train_df.head()


train_df.info()


# 1. Target Visualization
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='diagnosed_diabetes', palette='viridis')
plt.title("Distribution of Diabetic vs. Non-Diabetic Patients", fontsize=15, fontweight='bold')
plt.xlabel("Diagnosed Diabetes (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()




# 2. Correlation Heatmap (Numerical Features)
# We select only numeric columns for the heatmap
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
corr_matrix = train_df[numeric_cols].corr()

plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Hide the upper triangle (redundant)
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Matrix", fontsize=18, fontweight='bold')
plt.show()



# 3. Key Feature Analysis (BMI vs Diabetes)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=train_df, x='bmi', hue='diagnosed_diabetes', fill=True, common_norm=False, palette='crest')
plt.title("BMI Distribution by Diabetes Status", fontsize=15)
plt.show()


def preprocess_data(df, is_train=True):
    df_eng = df.copy()
    
    # -------------------------------------------------------
    # A. Feature Engineering (Domain Knowledge)
    # -------------------------------------------------------
    # Pulse Pressure: Indicator of arterial stiffness
    df_eng['Pulse_Pressure'] = df_eng['systolic_bp'] - df_eng['diastolic_bp']
    
    # Non-HDL Cholesterol: Often a better risk predictor than Total Cholesterol
    df_eng['Non_HDL'] = df_eng['cholesterol_total'] - df_eng['hdl_cholesterol']
    
    # Interaction: BMI is worse with Age
    df_eng['BMI_Age'] = df_eng['bmi'] * df_eng['age']
    
    # -------------------------------------------------------
    # B. Ordinal Encoding (Ranking Categories)
    # -------------------------------------------------------
    # We manually map these because the order matters!
    edu_map = {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3}
    inc_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
    
    df_eng['education_level'] = df_eng['education_level'].map(edu_map)
    df_eng['income_level'] = df_eng['income_level'].map(inc_map)
    
    # -------------------------------------------------------
    # C. One-Hot Encoding (Nominal Categories)
    # -------------------------------------------------------
    # We use pd.get_dummies to turn strings into numbers (0/1)
    categorical_cols = ['gender', 'ethnicity', 'smoking_status', 'employment_status']
    df_eng = pd.get_dummies(df_eng, columns=categorical_cols, drop_first=True)
    
    return df_eng

# Apply the function
# Note: We join Train and Test momentarily to ensure One-Hot Encoding creates same columns for both
train_len = len(train_df)
combined_df = pd.concat([train_df.drop('diagnosed_diabetes', axis=1), test_df], axis=0)

combined_processed = preprocess_data(combined_df)

# Split back into Train and Test
X = combined_processed.iloc[:train_len]
X_test = combined_processed.iloc[train_len:]
y = train_df['diagnosed_diabetes']

print("Data Preprocessing Complete.")
print(f"New Feature Count: {X.shape[1]}")


X.head()


y.head()


# Configuration for the "Winning" Model
# These parameters are tuned for large, balanced datasets like yours.
xgb_params = {
    "learning_rate": 0.010101790233963715, 
"max_depth": 4, 
"min_child_weight": 7.875908100225339, 
"subsample": 0.7225393932188394, 
"colsample_bytree": 0.5325708121965714, 
"gamma": 1.2582788478340508,
"lambda": 0.016947240752074988, 
"alpha": 7.335937487680093, 
"max_leaves": 123,
"booster": "gbtree",
"random_state":42,
"use_label_encoder": False,
"verbosity": 0,
"tree_method": "gpu_hist",
"predictor": "gpu_predictor",
"grow_policy": "lossguide",
"n_jobs": -1,
"n_estimators": 20000,
"max_bin": 256,
"objective": "binary:logistic",
"eval_metric": "auc"              # Faster for large datasets
}

# Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])
auc_scores = []

print("Starting Training Loop...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**xgb_params)
    
    # Train with Early Stopping
    # Stop if validation score doesn't improve for 50 rounds
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
        early_stopping_rounds=50
    )
    
    # Predict Probability (for AUC)
    val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Store results
    oof_preds[val_idx] = val_pred_proba
    test_preds += test_pred_proba / kf.get_n_splits() # Average predictions across folds
    
    fold_auc = roc_auc_score(y_val_fold, val_pred_proba)
    auc_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

print(f"\nAverage CV AUC: {np.mean(auc_scores):.5f}")


# Create Submission DataFrame
submission = pd.DataFrame({
    'id': test_id,
    'diagnosed_diabetes': test_preds
})

# Quick Sanity Check
print(submission.head())

# Save
submission.to_csv('submission.csv', index=False)
print("\nSuccess! 'submission.csv' is ready for upload.")

