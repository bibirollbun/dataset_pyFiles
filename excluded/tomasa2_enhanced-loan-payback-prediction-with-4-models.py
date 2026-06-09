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


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Set some display options for better visualization
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("Libraries imported successfully!")


# Load the data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
#df_orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

print(f"Training data shape: {df_train.shape}")
print(f"Test data shape: {df_test.shape}")
#print(f"Original dataset shape: {df_orig.shape}")
print(f"\nTarget distribution:")
print(df_train['loan_paid_back'].value_counts(normalize=True))


print("Training Data Head:")
df_train.head()


print("\nTraining Data Info:")
df_train.info()


print("\nMissing Values in Train Data:")
print(df_train.isnull().sum())


print("\nMissing Values in Test Data:")
print(df_test.isnull().sum())


# Descriptive statistics for numerical columns
df_train.describe()


# Distribution of the target variable 'accident_risk'
plt.figure(figsize=(10, 6))
sns.countplot(x='loan_paid_back', data=df_train, palette='pastel', edgecolor='black')
plt.title('Distribution of Loan Payback')
plt.xlabel('Loan Payback')
plt.ylabel('Count')
plt.show()


categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


# A more compact view of categorical features vs the target
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9, 0.66, 0.33])
target = 'loan_paid_back'

for i, col in enumerate(categorical_features):
    grouped = df_train.groupby(col)[target].mean()
    axes[i].bar(grouped.index.astype(str), grouped.values, color=colors)
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()


numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'loan_paid_back']]
print(numerical_features)


# Loop through all numerical features
for col in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # --- Left: Distribution (Histogram + KDE) ---
    sns.histplot(df_train[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f"Distribution of {col}", fontsize=12)
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequency")
    
    # --- Right: Boxplot (Outliers) ---
    sns.boxplot(x=df_train[col], ax=axes[1], color='lightcoral')
    axes[1].set_title(f"Boxplot of {col}", fontsize=12)
    axes[1].set_xlabel(col)
    
    # Clean layout
    plt.tight_layout()
    plt.show()



skew_values_train = df_train[numerical_features].apply(lambda x: skew(x.dropna()))
skew_values_test = df_test[numerical_features].apply(lambda x: skew(x.dropna()))
print('Skeweness in df_train')
print(skew_values_train.sort_values(ascending=False))
print('\n')
print('Skeweness in df_test')

print(skew_values_test.sort_values(ascending=False))


# #Lightgbm anf xgboost should deal with skewness and outliers just fine 
# # dealing with skeweness
# skewed_cols = skew_values_train[abs(skew_values_train) > 1].index.tolist()
# print("Highly skewed columns:", skewed_cols)

# for col in skewed_cols:
#     df_train[col] = np.log1p(df_train[col])
#     df_test[col]  = np.log1p(df_test[col])


# # dealing with outliers 
# for col in numerical_features:
#     lower = df_train[col].quantile(0.01)
#     upper = df_train[col].quantile(0.99)
#     df_train[col] = df_train[col].clip(lower, upper)
#     df_test[col]  = df_test[col].clip(lower, upper)


# # Specific feature engineering
# df_train['subgrade'] = df_train['grade_subgrade'].str[1:].astype(int)
# df_test['subgrade'] = df_test['grade_subgrade'].str[1:].astype(int)

# df_train['grade'] = df_train['grade_subgrade'].str[0]
# df_test['grade'] = df_test['grade_subgrade'].str[0]


target_col = 'loan_paid_back'


combined_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)


def create_minimal_features(df):
    """Keep only grade_rank - the ONE engineered feature that helps"""
    
    # Only this one - it has 2.92 importance, decent contribution
    df['grade_letter'] = df['grade_subgrade'].str[0]
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_rank'] = df['grade_letter'].map(grade_map)
    
    return df


combined_df = create_minimal_features(combined_df)

#columns_to_remove = ['grade_subgrade','grade_letter']
columns_to_remove = ['grade_letter']
#combined_df.drop(columns=columns_to_remove, inplace = True)

df_train = combined_df.iloc[:len(df_train)]
df_test = combined_df.iloc[len(df_train):].drop(columns=[target_col]).reset_index(drop=True)


def create_frequency_features(df, df_test, cat_cols, num_cols):
    """
    Add frequency and binning features efficiently.

    - For each categorical column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5, 10, 15 quantile bins.

    Parameters:
        df (pd.DataFrame): Training DataFrame
        df_test (pd.DataFrame): Test DataFrame
        cat_cols (list): Categorical columns for frequency encoding
        num_cols (list): Numerical columns for quantile binning    
    """
    # Pre-allocate result DataFrames
    freq_train = pd.DataFrame(index=df.index)
    freq_test = pd.DataFrame(index=df_test.index)
    bin_train = pd.DataFrame(index=df.index)
    bin_test = pd.DataFrame(index=df_test.index)

    # --------- Frequency Encoding ---------
    for col in cat_cols:
        freq = df[col].value_counts()

        freq_train[f"{col}_freq"] = df[col].map(freq).astype("float32")
        freq_test[f"{col}_freq"] = df_test[col].map(freq).fillna(0).astype("float32")

    # --------- Quantile Binning (5, 10, 15) ---------
    quantile_list = [5, 10, 15]

    for col in num_cols:
        col_train = df[col]
        col_test = df_test[col]

        for q in quantile_list:
            try:
                train_bins, bins = pd.qcut(
                    col_train,
                    q=q,
                    labels=False,
                    retbins=True,
                    duplicates="drop"
                )

                bin_train[f"{col}_bin{q}"] = train_bins.astype("int8")

                # Apply bins to test using pd.cut
                test_bins = pd.cut(
                    col_test,
                    bins=bins,
                    labels=False,
                    include_lowest=True
                ).fillna(0)

                bin_test[f"{col}_bin{q}"] = test_bins.astype("int8")

            except Exception:
                # Column not suitable for quantile splitting
                bin_train[f"{col}_bin{q}"] = 0
                bin_test[f"{col}_bin{q}"] = 0

    # --------- Add features to df / df_test ---------
    df = pd.concat([df, freq_train, bin_train], axis=1)
    df_test = pd.concat([df_test, freq_test, bin_test], axis=1)

    return df, df_test



def create_binned_interactions(df):
    """
    Create interactions between selected 10-quantile binned features 
    and key categorical features.
    This keeps the original risk logic but adapts to the new naming.
    """

    # Credit quality × Employment
    df['credit_emp'] = (df['credit_score_bin10'].astype(str) + '_' +
                        df['employment_status'].astype(str))

    # DTI × Grade
    df['debt_to_income_ratio_grade'] = (df['debt_to_income_ratio_bin10'].astype(str) + '_' +
                                        df['grade_subgrade'].astype(str))

    # Interest rate × Employment
    df['interest_rate_emp'] = (df['interest_rate_bin10'].astype(str) + '_' +
                               df['employment_status'].astype(str))

    # Loan size × Income
    df['loan_income_cross'] = (df['loan_amount_bin10'].astype(str) + '_' +
                               df['annual_income_bin10'].astype(str))

    return df



df_train, df_test = create_frequency_features(df_train, df_test, 
                                              cat_cols=categorical_features, 
                                              num_cols=numerical_features)

df_train = create_binned_interactions(df_train)
df_test = create_binned_interactions(df_test)


df_train


df_test


categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


import category_encoders as ce


print("Target encoding will be performed within cross-validation to prevent data leakage")


# Save the target variable and IDs
y = df_train['loan_paid_back']
test_ids = df_test['id']

df_train_processed = df_train.drop(['id', 'loan_paid_back'], axis=1)
df_test_processed = df_test.drop(['id'], axis=1)

X = df_train_processed.copy()
X_test = df_test_processed.copy()

print(f"Training data shape before correlation: {X.shape}")
print(f"Test data shape before correlation: {X_test.shape}")


numerical_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'loan_paid_back']]
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


# print("Adding features from external original dataset...")
# global_orig_mean = df_orig['loan_paid_back'].mean()
# orig_feature_names = []

# # ============================================
# # ONLY for CATEGORICAL features 
# # ============================================
# for col in categorical_features:  # ← Changed from orig_cols
#     if col not in df_orig.columns:
#         continue
    
#     # Smoothed mean target encoding
#     mean_col = f"orig_mean_{col}"
#     smoothing = 10
#     mean_map = df_orig.groupby(col)['loan_paid_back'] \
#                       .apply(lambda x: (x.mean() * len(x) + global_orig_mean * smoothing) /
#                                        (len(x) + smoothing))
    
#     combined_df[mean_col] = combined_df[col].map(mean_map)
#     combined_df[mean_col] = combined_df[mean_col].fillna(global_orig_mean)
#     orig_feature_names.append(mean_col)
    
#     # Count encoding (fixed)
#     count_col = f"orig_count_{col}"
#     count_map = df_orig.groupby(col).size()  # ← Removed ['loan_paid_back']
#     combined_df[count_col] = combined_df[col].map(count_map)
#     combined_df[count_col] = combined_df[count_col].fillna(0)
#     orig_feature_names.append(count_col)

# # ============================================  
# # For NUMERICAL features - use percentiles instead
# # ============================================
# for col in numerical_features:  # ← Added separate loop
#     if col not in df_orig.columns:
#         continue
    
#     # Percentile feature (less sparse than groupby)
#     pct_col = f"orig_pct_{col}"
#     combined_df[pct_col] = combined_df[col].apply(
#         lambda x: (df_orig[col] <= x).mean() * 100 if pd.notna(x) else 50
#     )
#     orig_feature_names.append(pct_col)

# print(f"Added {len(orig_feature_names)} orig-based features.")




# # Remove highly correlated features (threshold = 0.85)
# corr_matrix = combined_df.select_dtypes(include=[np.number]).corr().abs()
# corr_matrix = corr_matrix.fillna(0)

# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# threshold = 0.85
# to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# print(f"Number of features to drop due to correlation > {threshold}: {len(to_drop)}")
# print("Dropped features:", to_drop)

# # Create filtered dataframe with dropped columns
# combined_df_filtered = combined_df.drop(columns=to_drop)

# # because I created new catagorical features let make new updated list 
# categorical_features = combined_df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()

# #categorical_features = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()

# # IMPORTANT: propagate the dropped-columns filtering back to df_train_fe and df_test_fe
# # so that later target-encoding inside CV and model training use the same feature set.
# # This prevents mismatches between the features used to create CV splits and the features
# # actually used during training.

# df_train_fe = combined_df_filtered.iloc[:len(df_train)].reset_index(drop=True)
# df_test_fe = combined_df_filtered.iloc[len(df_train):].reset_index(drop=True)

# # Separate back into training and testing sets (for diagnostic printing)
# X = combined_df_filtered.iloc[:len(df_train)]
# X_test = combined_df_filtered.iloc[len(df_train):]

# print(f"Final training data shape after correlation: {X.shape}")
# print(f"Final test data shape after correlation: {X_test.shape}")


categorical_features


# # Correlation heatmap for numerical features
# numerical_features = combined_df_filtered.select_dtypes(include=np.number)
# # id is not important for correlation matrix
# numerical_features.drop(columns=['id'], errors='ignore', inplace=True)
# plt.figure(figsize=(12, 10))
# sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix of AFTER Threshold and Target Encoding')
# plt.show()


# # XGBoost LGB and CAT parameters 
# # OPTUNA for later ??
# # in new vesrion I made some changes to parameters to see if it will improve overal AUC 
# # Optuna takes too much time so I will manually change parameters and I will learn during process what they actually do 
# xgb_params = {
#     'objective': 'binary:logistic',
#     "device": "cuda",          # lets try GPU
#     'eval_metric': 'auc',
#     'n_estimators': 4000,      # more trees    2000
#     'early_stopping_rounds': 100,
#     'tree_method': 'hist',
#     'n_jobs': -1,
#     'random_state': 42,
#     'learning_rate': 0.02,     #smaller LR
#     'max_depth': 3,            # safer from overfitting     4
#     'subsample': 0.75,         # stronger regularization   0.99
#     'colsample_bytree': 0.55,  # more column randomness    0.68
#     'gamma': 1,                # reduce overfitting
#     'reg_lambda': 1.0,         # stronger L2            0.0036
#     'reg_alpha': 0.1           # slightly stronger L1   0.021
# }

# # LightGBM parameters
# lgb_params = {
#     'objective': 'binary',
#     'device': 'gpu',            # lets try GPU
#     'metric': 'auc',
#     'boosting_type': 'gbdt',
#     'n_estimators': 4000,       # number of trees   2000
#     'early_stopping_rounds': 100,
#     'learning_rate': 0.015,     # small learning rate with high number tree is better 0.03    
#     'num_leaves': 31,           # shallower tree → higher AUC stability   50
#     'max_depth': -1,            # let leaves control complexity           6
#     'subsample_freq': 1, 
#     'min_child_samples': 40,    # safer split constraint                  20
#     'subsample': 0.75,          # 0.8
#     'colsample_bytree': 0.65,    # 0.8
#     'reg_alpha': 0.4,          # 0.05
#     'reg_lambda': 0.6,         # 0.01
#     'random_state': 42,
#     'n_jobs': -1,
#     'verbose': -1
# }

# # CatBoost parameters
# catboost_params = {
#     'task_type': 'GPU',        # lets try GPU
#     'devices': '0',            # lets try GPU
#     'iterations': 2500,        # 2000
#     'learning_rate': 0.02,     # learnin rate too high  0.05
#     'depth': 6,                # too low ?  5
#     'l2_leaf_reg': 5,          # 3
#     'loss_function': 'Logloss',
#     'eval_metric': 'AUC',
#     'random_seed': 42,
#     'od_type': 'Iter',
#     'od_wait': 100,
#     'grow_policy': 'Lossguide',
#     'verbose': False,
#     'allow_writing_files': False,
#     'verbose': False
# }

# # Random Forest parameters
# rf_params = {
#     'n_estimators': 500,
#     'max_depth': 10,
#     'min_samples_split': 20,
#     'min_samples_leaf': 10,
#     'max_features': 'sqrt',
#     'random_state': 42,
#     'n_jobs': -1,
#     'class_weight': 'balanced_subsample'
# }

# print("Model parameters defined successfully!")


# XGBoost parameters 
# OPTUNA for later ??
xgb_params = {
    'device': 'cuda',          # lets try GPU
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': 42,
    'learning_rate': 0.1,
    'max_depth': 4,
    'subsample': 0.99,
    'colsample_bytree': 0.68,
    'gamma': 8.9e-06,
    'reg_lambda': 0.0036,
    'reg_alpha': 0.021
}

# LightGBM parameters
# lgb_params = {
#     'device': 'gpu',            # lets try GPU
#     'objective': 'binary',
#     'metric': 'auc',
#     'boosting_type': 'gbdt',
#     'n_estimators': 2000,
#     'early_stopping_rounds': 100,
#     'learning_rate': 0.05,
#     'num_leaves': 31,
#     'max_depth': 5,
#     'min_child_samples': 20,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'reg_alpha': 0.01,
#     'reg_lambda': 0.01,
#     'random_state': 42,
#     'n_jobs': -1,
#     'verbose': -1
# }

#lgb_params = {
#    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
#    'max_depth': 6, 'num_leaves': 50, 'learning_rate': 0.03,
#    'colsample_bytree': 0.8, 'subsample': 0.8,
#    'subsample_freq': 1, 'min_child_samples': 20, 'reg_alpha': 0.05,
#    'reg_lambda': 0.1, 'random_state': 42,'n_estimators': 2000,
#    'n_jobs': -1, 
#    'device': 'gpu',
#    'verbose': -1,
#}

lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'max_depth': 6, 'num_leaves': 50, 'learning_rate': 0.01,
    'colsample_bytree': 0.8, 'subsample': 0.8,
    'subsample_freq': 1, 'min_child_samples': 20, 'reg_alpha': 0.05,
    'reg_lambda': 0.1, 'random_state': 42,
    'n_jobs': -1, 'device': 'gpu','verbose': -1,'n_estimators': 2000,
    'early_stopping_rounds': 100,
}

# CatBoost parameters
# catboost_params = {
#     'task_type': 'GPU',        # lets try GPU
#     'devices': '0',            # lets try GPU
#     'metric_period': 5,        # lets try GPU it will not make a warning each loop  common performance warning, not an error. It's telling you that AUC can't be calculated on the GPU,
#     'iterations': 2000,
#     'learning_rate': 0.05,
#     'depth': 5,
#     'l2_leaf_reg': 3,
#     'loss_function': 'Logloss',
#     'eval_metric': 'AUC',
#     'random_seed': 42,
#     'od_type': 'Iter',
#     'od_wait': 100,
#     'grow_policy': 'Lossguide',
#     'verbose': False,
#     'allow_writing_files': False
# }

catboost_params = {
    'iterations': 4000,
    'learning_rate': 0.05,
    'depth': 5,
    'l2_leaf_reg': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'od_type': 'Iter',      # Use early stopping
    'od_wait': 100,         # Stop after 100 rounds of no improvement
    'grow_policy': 'Lossguide',
    'thread_count': -1,
    'verbose': False,       # Suppress training output
    'verbose': 100,
    'allow_writing_files': False
}


# Random Forest parameters
rf_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced_subsample'
}

print("Model parameters defined successfully!")


# import optuna
# from optuna.samplers import TPESampler
# import warnings
# warnings.filterwarnings('ignore')

# # Set up Optuna logging
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# def objective_xgboost(trial, X_train_full, y_train_full, categorical_features, n_folds=3):
#     """Objective function for XGBoost optimization"""
    
#     # Suggest hyperparameters
#     params = {
#         'objective': 'binary:logistic',
#         'device': 'cuda',
#         'eval_metric': 'auc',
#         'tree_method': 'hist',
#         'n_jobs': -1,
#         'random_state': 42,
        
#         # Parameters to optimize
#         'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=100),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10, log=True),
#         'early_stopping_rounds': 50
#     }
    
#     # Internal CV for Optuna
#     skf_inner = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
#     cv_scores = []
    
#     for train_idx, val_idx in skf_inner.split(X_train_full, y_train_full):
#         X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
#         y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
#         # Apply target encoding
#         target_encoder = ce.TargetEncoder(cols=categorical_features)
#         target_encoder.fit(X_train[categorical_features], y_train)
        
#         X_train_encoded = X_train.copy()
#         X_val_encoded = X_val.copy()
#         X_train_encoded[categorical_features] = target_encoder.transform(X_train[categorical_features])
#         X_val_encoded[categorical_features] = target_encoder.transform(X_val[categorical_features])
        
#         # Train model
#         model = xgb.XGBClassifier(**params)
#         model.fit(
#             X_train_encoded, y_train,
#             eval_set=[(X_val_encoded, y_val)],
#             verbose=False
#         )
        
#         # Predict and calculate AUC
#         val_preds = model.predict_proba(X_val_encoded)[:, 1]
#         auc = roc_auc_score(y_val, val_preds)
#         cv_scores.append(auc)
    
#     return np.mean(cv_scores)

# def objective_lightgbm(trial, X_train_full, y_train_full, categorical_features, n_folds=3):
#     """Objective function for LightGBM optimization"""
    
#     params = {
#         'objective': 'binary',
#         'device': 'gpu',
#         'metric': 'auc',
#         'boosting_type': 'gbdt',
#         'random_state': 42,
#         'n_jobs': -1,
#         'verbose': -1,
        
#         # Parameters to optimize
#         'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=100),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 300),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10, log=True),
#         'early_stopping_rounds': 50
#     }
    
#     # Internal CV for Optuna
#     skf_inner = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
#     cv_scores = []
    
#     for train_idx, val_idx in skf_inner.split(X_train_full, y_train_full):
#         X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
#         y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
#         # Apply target encoding
#         target_encoder = ce.TargetEncoder(cols=categorical_features)
#         target_encoder.fit(X_train[categorical_features], y_train)
        
#         X_train_encoded = X_train.copy()
#         X_val_encoded = X_val.copy()
#         X_train_encoded[categorical_features] = target_encoder.transform(X_train[categorical_features])
#         X_val_encoded[categorical_features] = target_encoder.transform(X_val[categorical_features])
        
#         # Train model
#         model = lgb.LGBMClassifier(**params)
#         model.fit(
#             X_train_encoded, y_train,
#             eval_set=[(X_val_encoded, y_val)],
#             callbacks=[lgb.log_evaluation(0)]
#         )
        
#         val_preds = model.predict_proba(X_val_encoded)[:, 1]
#         auc = roc_auc_score(y_val, val_preds)
#         cv_scores.append(auc)
    
#     return np.mean(cv_scores)

# def objective_catboost(trial, X_train_full, y_train_full, categorical_features, n_folds=3):
#     """Objective function for CatBoost optimization"""
    
#     params = {
#         'task_type': 'GPU',
#         'devices': '0',
#         'loss_function': 'Logloss',
#         'eval_metric': 'AUC',
#         'random_seed': 42,
#         'od_type': 'Iter',
#         'verbose': False,
#         'allow_writing_files': False,
#         'metric_period': 5,
        
#         # Parameters to optimize
#         'iterations': trial.suggest_int('iterations', 100, 3000, step=100),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'depth': trial.suggest_int('depth', 4, 10),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#         'border_count': trial.suggest_int('border_count', 32, 255),
#         'od_wait': trial.suggest_int('od_wait', 20, 100)
#     }
    
#     # Add grow_policy conditionally
#     if params['depth'] > 6:
#         params['grow_policy'] = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
    
#     # Internal CV for Optuna
#     skf_inner = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
#     cv_scores = []
    
#     for train_idx, val_idx in skf_inner.split(X_train_full, y_train_full):
#         X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
#         y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
#         # Apply target encoding
#         target_encoder = ce.TargetEncoder(cols=categorical_features)
#         target_encoder.fit(X_train[categorical_features], y_train)
        
#         X_train_encoded = X_train.copy()
#         X_val_encoded = X_val.copy()
#         X_train_encoded[categorical_features] = target_encoder.transform(X_train[categorical_features])
#         X_val_encoded[categorical_features] = target_encoder.transform(X_val[categorical_features])
        
#         # Train model
#         model = cb.CatBoostClassifier(**params)
#         model.fit(
#             X_train_encoded, y_train,
#             eval_set=(X_val_encoded, y_val),
#             use_best_model=True,
#             verbose=False
#         )
        
#         val_preds = model.predict_proba(X_val_encoded)[:, 1]
#         auc = roc_auc_score(y_val, val_preds)
#         cv_scores.append(auc)
    
#     return np.mean(cv_scores)

# # Function to run optimization and get best params
# def optimize_model(model_name, X_train, y_train, categorical_features, n_trials=100):
#     """Run Optuna optimization for a specific model"""
    
#     print(f"\n{'='*60}")
#     print(f"Optimizing {model_name.upper()} with Optuna")
#     print(f"{'='*60}")
    
#     study = optuna.create_study(
#         direction='maximize',
#         sampler=TPESampler(seed=42)
#     )
    
#     if model_name == 'xgboost':
#         objective = lambda trial: objective_xgboost(trial, X_train, y_train, categorical_features)
#     elif model_name == 'lightgbm':
#         objective = lambda trial: objective_lightgbm(trial, X_train, y_train, categorical_features)
#     elif model_name == 'catboost':
#         objective = lambda trial: objective_catboost(trial, X_train, y_train, categorical_features)
#     else:
#         raise ValueError(f"Unknown model: {model_name}")
    
#     study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
#     print(f"\nBest trial:")
#     print(f"  AUC: {study.best_value:.5f}")
#     print(f"  Params: {study.best_params}")
    
#     return study.best_params

# # MAIN TRAINING PIPELINE WITH OPTUNA
# # First, optimize hyperparameters using a subset of data (optional for speed)
# OPTIMIZE_PARAMS = True  # Set to False to use your manual parameters
# N_OPTUNA_TRIALS = 100   # Number of trials for each model (increase for better results)

# if OPTIMIZE_PARAMS:
#     # Use a subset for faster optimization (optional)
#     sample_size = min(100000, len(df_train_fe))
#     sample_idx = np.random.choice(df_train_fe.index, sample_size, replace=False)
#     X_sample = df_train_fe.loc[sample_idx]
#     y_sample = y.loc[sample_idx]
    
#     # Optimize each model
#     best_params = {}
    
#     # XGBoost
#     xgb_best = optimize_model('xgboost', X_sample, y_sample, categorical_features, N_OPTUNA_TRIALS)
#     xgb_params.update(xgb_best)
#     best_params['xgboost'] = xgb_best
    
#     # LightGBM
#     lgb_best = optimize_model('lightgbm', X_sample, y_sample, categorical_features, N_OPTUNA_TRIALS)
#     lgb_params.update(lgb_best)
#     best_params['lightgbm'] = lgb_best
    
#     # CatBoost
#     cb_best = optimize_model('catboost', X_sample, y_sample, categorical_features, N_OPTUNA_TRIALS)
#     catboost_params.update(cb_best)
#     best_params['catboost'] = cb_best
    
#     # Save best parameters for future use
#     import json
#     with open('best_params.json', 'w') as f:
#         json.dump(best_params, f, indent=2)
    
#     print("\n" + "="*60)
#     print("OPTIMIZATION COMPLETE - Best Parameters Saved")
#     print("="*60)

# # Now run your original training loop with optimized parameters
# # (Your existing training code continues here...)


catboost_params


xgb_params


lgb_params


# Setup Stratifield K-Fold Cross-Validation
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Initialize storage for out-of-fold predictions and test predictions
oof_predictions = {
    'xgboost': np.zeros(X.shape[0]),
    'lightgbm': np.zeros(X.shape[0]),
    'catboost': np.zeros(X.shape[0]),
    'random_forest': np.zeros(X.shape[0])
}

test_predictions = {
    'xgboost': np.zeros(X_test.shape[0]),
    'lightgbm': np.zeros(X_test.shape[0]),
    'catboost': np.zeros(X_test.shape[0]),
    'random_forest': np.zeros(X_test.shape[0])
}

# Store models for each fold
models = {
    'xgboost': [],
    'lightgbm': [],
    'catboost': [],
    'random_forest': []
}

# Store AUC scores for each model
auc_scores = {
    'xgboost': [],
    'lightgbm': [],
    'catboost': [],
    'random_forest': []
}



for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*60}")
    
    # Get fold data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # =====================
    # TARGET ENCODING for XGBoost, LightGBM, Random Forest
    # =====================
    target_encoder = ce.TargetEncoder(cols=categorical_features)
    target_encoder.fit(X_train[categorical_features], y_train)
    
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()
    
    X_train_encoded[categorical_features] = target_encoder.transform(X_train[categorical_features])
    X_val_encoded[categorical_features] = target_encoder.transform(X_val[categorical_features])
    X_test_encoded[categorical_features] = target_encoder.transform(X_test[categorical_features])
    
    # =====================
    # XGBoost
    # =====================
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_encoded, y_train,
        eval_set=[(X_val_encoded, y_val)],
        verbose=False
    )
    val_preds = xgb_model.predict_proba(X_val_encoded)[:, 1]
    oof_predictions['xgboost'][val_idx] = val_preds
    test_predictions['xgboost'] += xgb_model.predict_proba(X_test_encoded)[:, 1] / N_SPLITS
    auc = roc_auc_score(y_val, val_preds)
    auc_scores['xgboost'].append(auc)
    print(f"XGBoost Fold {fold+1} AUC: {auc:.5f}")
    models['xgboost'].append(xgb_model)
    
    # =====================
    # LightGBM
    # =====================
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train_encoded, y_train,
        eval_set=[(X_val_encoded, y_val)],
        callbacks=[lgb.log_evaluation(0)]
    )
    val_preds = lgb_model.predict_proba(X_val_encoded)[:, 1]
    oof_predictions['lightgbm'][val_idx] = val_preds
    test_predictions['lightgbm'] += lgb_model.predict_proba(X_test_encoded)[:, 1] / N_SPLITS
    auc = roc_auc_score(y_val, val_preds)
    auc_scores['lightgbm'].append(auc)
    print(f"LightGBM Fold {fold+1} AUC: {auc:.5f}")
    models['lightgbm'].append(lgb_model)
    
    # =====================
    # CatBoost - USE NATIVE CATEGORICAL HANDLING
    # =====================
    print("\nTraining CatBoost...")
    cb_model = cb.CatBoostClassifier(**catboost_params)
    cb_model.fit(
        X_train, y_train,  # Use ORIGINAL data, not encoded
        eval_set=(X_val, y_val),
        cat_features=categorical_features,  # Specify categorical features
        #use_best_model=True,
        verbose=False
    )
    val_preds = cb_model.predict_proba(X_val)[:, 1]
    oof_predictions['catboost'][val_idx] = val_preds
    test_predictions['catboost'] += cb_model.predict_proba(X_test)[:, 1] / N_SPLITS  # Original X_test
    auc = roc_auc_score(y_val, val_preds)
    auc_scores['catboost'].append(auc)
    print(f"CatBoost Fold {fold+1} AUC: {auc:.5f}")
    models['catboost'].append(cb_model)
    
    # =====================
    # Random Forest
    # =====================
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train_encoded, y_train)
    val_preds = rf_model.predict_proba(X_val_encoded)[:, 1]
    oof_predictions['random_forest'][val_idx] = val_preds
    test_predictions['random_forest'] += rf_model.predict_proba(X_test_encoded)[:, 1] / N_SPLITS
    auc = roc_auc_score(y_val, val_preds)
    auc_scores['random_forest'].append(auc)
    print(f"Random Forest Fold {fold+1} AUC: {auc:.5f}")
    models['random_forest'].append(rf_model)


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*60}")
    
    # Get fold data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialize target encoder for this fold
    target_encoder = ce.TargetEncoder(cols=categorical_features)
    
    # Fit encoder on training data only
    target_encoder.fit(X_train[categorical_features], y_train)
    
    # Transform train, validation, and test data using the same encoder
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()
    
    X_train_encoded[categorical_features] = target_encoder.transform(X_train[categorical_features])
    X_val_encoded[categorical_features] = target_encoder.transform(X_val[categorical_features])
    X_test_encoded[categorical_features] = target_encoder.transform(X_test[categorical_features])
    
    # =====================
    # XGBoost
    # =====================
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_encoded, y_train,
        eval_set=[(X_val_encoded, y_val)],
        verbose=False
    )
    val_preds = xgb_model.predict_proba(X_val_encoded)[:, 1]
    oof_predictions['xgboost'][val_idx] = val_preds
    test_predictions['xgboost'] += xgb_model.predict_proba(X_test_encoded)[:, 1] / N_SPLITS
    auc = roc_auc_score(y_val, val_preds)
    auc_scores['xgboost'].append(auc)
    print(f"XGBoost Fold {fold+1} AUC: {auc:.5f}")
    models['xgboost'].append(xgb_model)
    
    # =====================
    # LightGBM
    # =====================
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train_encoded, y_train,
        eval_set=[(X_val_encoded, y_val)],
        callbacks=[lgb.log_evaluation(0)]
    )
    val_preds = lgb_model.predict_proba(X_val_encoded)[:, 1]
    oof_predictions['lightgbm'][val_idx] = val_preds
    test_predictions['lightgbm'] += lgb_model.predict_proba(X_test_encoded)[:, 1] / N_SPLITS
    auc = roc_auc_score(y_val, val_preds)
    auc_scores['lightgbm'].append(auc)
    print(f"LightGBM Fold {fold+1} AUC: {auc:.5f}")
    models['lightgbm'].append(lgb_model)
    
    # =====================
    # CatBoost
    # =====================
    print("\nTraining CatBoost...")
    cb_model = cb.CatBoostClassifier(**catboost_params)
    cb_model.fit(
        X_train_encoded, y_train,
        eval_set=(X_val_encoded, y_val),
        use_best_model=True,
        verbose=False
    )
    val_preds = cb_model.predict_proba(X_val_encoded)[:, 1]
    oof_predictions['catboost'][val_idx] = val_preds
    test_predictions['catboost'] += cb_model.predict_proba(X_test_encoded)[:, 1] / N_SPLITS
    auc = roc_auc_score(y_val, val_preds)
    auc_scores['catboost'].append(auc)
    print(f"CatBoost Fold {fold+1} AUC: {auc:.5f}")
    models['catboost'].append(cb_model)
    
    # =====================
    # Random Forest
    # =====================
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train_encoded, y_train)
    val_preds = rf_model.predict_proba(X_val_encoded)[:, 1]
    oof_predictions['random_forest'][val_idx] = val_preds
    test_predictions['random_forest'] += rf_model.predict_proba(X_test_encoded)[:, 1] / N_SPLITS
    auc = roc_auc_score(y_val, val_preds)
    auc_scores['random_forest'].append(auc)
    print(f"Random Forest Fold {fold+1} AUC: {auc:.5f}")
    models['random_forest'].append(rf_model)


# Print CV results for each model
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS")
print("="*60)

for model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
    mean_auc = np.mean(auc_scores[model_name])
    std_auc = np.std(auc_scores[model_name])
    print(f"\n{model_name.upper()}:")
    print(f"  Average CV AUC: {mean_auc:.5f} (+/- {std_auc:.5f})")
    print(f"  Fold scores: {[f'{score:.5f}' for score in auc_scores[model_name]]}")


# # Meta features from OOF predictions
# p_xgb = oof_predictions['xgboost']
# p_lgb = oof_predictions['lightgbm']
# p_cb  = oof_predictions['catboost']

# X_meta = np.column_stack([p_xgb, p_lgb, p_cb

# from sklearn.linear_model import LogisticRegression

# meta = LogisticRegression(max_iter=1000)
# meta.fit(X_meta, y)

# xgb_test_pred = test_predictions['xgboost']
# lgb_test_pred = test_predictions['lightgbm']
# cb_test_pred  = test_predictions['catboost']

# X_meta_test = np.column_stack([
#     xgb_test_pred,
#     lgb_test_pred,
#     cb_test_pred
# ])

# final_prob = meta.predict_proba(X_meta_test)[:, 1]



# #calibration
# from sklearn.isotonic import IsotonicRegression

# iso = IsotonicRegression(out_of_bounds='clip')
# iso.fit(meta.predict_proba(X_meta)[:,1], y)

# final_prob = iso.predict(final_prob)


# # ============================================================
# # Final Submission for Stacking Meta-Model
# # ============================================================

# # Ensure predictions are within [0, 1] range
# final_predictions = np.clip(final_prob, 0, 1)

# # Create submission dataframe
# submission_df = pd.DataFrame({
#     'id': test_ids,
#     'loan_paid_back': final_predictions
# })

# # Save the submission file
# submission_df.to_csv("submission.csv", index=False)
# print("\nSubmission file created: submission.csv")

# # Show preview + statistics
# print("\nSample of final submission:")
# print(submission_df.head(10))

# print("\nPrediction statistics:")
# print(submission_df['loan_paid_back'].describe())



# Strategy 1: Simple Average Ensemble
print("\nEnsemble Strategy 1: Simple Average")
print("="*40)

ensemble_oof_simple = np.mean([
    oof_predictions['xgboost'],
    oof_predictions['lightgbm'],
    oof_predictions['catboost'],
    oof_predictions['random_forest']
], axis=0)

ensemble_test_simple = np.mean([
    test_predictions['xgboost'],
    test_predictions['lightgbm'],
    test_predictions['catboost'],
    test_predictions['random_forest']
], axis=0)

ensemble_auc_simple = roc_auc_score(y, ensemble_oof_simple)
print(f"Simple Average Ensemble CV AUC: {ensemble_auc_simple:.5f}")


# Strategy 2: Weighted Average Ensemble (based on CV performance)
print("\nEnsemble Strategy 2: Weighted Average (based on CV performance)")
print("="*40)

# Calculate weights based on CV AUC scores
model_weights = {}
total_auc = 0
for model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
    mean_auc = np.mean(auc_scores[model_name])
    model_weights[model_name] = mean_auc
    total_auc += mean_auc

# Normalize weights to sum to 1
for model_name in model_weights:
    model_weights[model_name] /= total_auc
    print(f"{model_name} weight: {model_weights[model_name]:.4f}")

# Create weighted ensemble predictions
ensemble_oof_weighted = (
    model_weights['xgboost'] * oof_predictions['xgboost'] +
    model_weights['lightgbm'] * oof_predictions['lightgbm'] +
    model_weights['catboost'] * oof_predictions['catboost'] +
    model_weights['random_forest'] * oof_predictions['random_forest']
)

ensemble_test_weighted = (
    model_weights['xgboost'] * test_predictions['xgboost'] +
    model_weights['lightgbm'] * test_predictions['lightgbm'] +
    model_weights['catboost'] * test_predictions['catboost'] +
    model_weights['random_forest'] * test_predictions['random_forest']
)

ensemble_auc_weighted = roc_auc_score(y, ensemble_oof_weighted)
print(f"\nWeighted Average Ensemble CV AUC: {ensemble_auc_weighted:.5f}")


# Strategy 3: Rank Average Ensemble
print("\nEnsemble Strategy 3: Rank Average")
print("="*40)

from scipy.stats import rankdata

# Rank the predictions for each model
oof_ranks = {
    model_name: rankdata(oof_predictions[model_name]) / len(oof_predictions[model_name])
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']
}

test_ranks = {
    model_name: rankdata(test_predictions[model_name]) / len(test_predictions[model_name])
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']
}

# Average the ranks
ensemble_oof_rank = np.mean(list(oof_ranks.values()), axis=0)
ensemble_test_rank = np.mean(list(test_ranks.values()), axis=0)

ensemble_auc_rank = roc_auc_score(y, ensemble_oof_rank)
print(f"Rank Average Ensemble CV AUC: {ensemble_auc_rank:.5f}")


# Strategy 4: Optimized Weighted Ensemble using scipy.optimize
print("\nEnsemble Strategy 4: Optimized Weighted Ensemble")
print("="*40)

from scipy.optimize import minimize

def ensemble_score(weights):
    """Calculate negative AUC score for optimization (minimize negative = maximize positive)"""
    ensemble_pred = (
        weights[0] * oof_predictions['xgboost'] +
        weights[1] * oof_predictions['lightgbm'] +
        weights[2] * oof_predictions['catboost'] +
        weights[3] * oof_predictions['random_forest']
    )
    return -roc_auc_score(y, ensemble_pred)

# Initial weights (equal)
init_weights = [0.25, 0.25, 0.25, 0.25]
#init_weights = [1/3, 1/3, 1/3]

# Constraints: weights sum to 1 and are between 0 and 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * 4

# Optimize
result = minimize(ensemble_score, init_weights, method='SLSQP', 
                 bounds=bounds, constraints=constraints)

optimal_weights = result.x
print(f"Optimal weights:")
print(f"  XGBoost: {optimal_weights[0]:.4f}")
print(f"  LightGBM: {optimal_weights[1]:.4f}")
print(f"  CatBoost: {optimal_weights[2]:.4f}")
print(f"  Random Forest: {optimal_weights[3]:.4f}")

# Create optimized ensemble predictions
ensemble_oof_optimized = (
    optimal_weights[0] * oof_predictions['xgboost'] +
    optimal_weights[1] * oof_predictions['lightgbm'] +
    optimal_weights[2] * oof_predictions['catboost'] +
    optimal_weights[3] * oof_predictions['random_forest']
)

ensemble_test_optimized = (
    optimal_weights[0] * test_predictions['xgboost'] +
    optimal_weights[1] * test_predictions['lightgbm'] +
    optimal_weights[2] * test_predictions['catboost'] +
    optimal_weights[3] * test_predictions['random_forest']
)

ensemble_auc_optimized = roc_auc_score(y, ensemble_oof_optimized)
print(f"\nOptimized Ensemble CV AUC: {ensemble_auc_optimized:.5f}")


# Summary of all model performances
print("\n" + "="*60)
print("FINAL PERFORMANCE COMPARISON")
print("="*60)

# Individual models
print("\nIndividual Models:")
print("-"*30)
for model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
    mean_auc = np.mean(auc_scores[model_name])
    print(f"{model_name.ljust(15)}: {mean_auc:.5f}")

# Ensemble strategies
print("\nEnsemble Strategies:")
print("-"*30)
print(f"Simple Average  : {ensemble_auc_simple:.5f}")
print(f"Weighted Average: {ensemble_auc_weighted:.5f}")
print(f"Rank Average    : {ensemble_auc_rank:.5f}")
print(f"Optimized       : {ensemble_auc_optimized:.5f}")

# Find best strategy
best_ensemble_scores = {
    'Simple': ensemble_auc_simple,
    'Weighted': ensemble_auc_weighted,
    'Rank': ensemble_auc_rank,
    'Optimized': ensemble_auc_optimized
}
best_strategy = max(best_ensemble_scores, key=best_ensemble_scores.get)
print(f"\nBest Ensemble Strategy: {best_strategy} with AUC = {best_ensemble_scores[best_strategy]:.5f}")


# Get feature importance from tree-based models
import matplotlib.pyplot as plt

# Average feature importance across folds for each model
feature_importance = pd.DataFrame()
feature_importance['feature'] = X.columns

# XGBoost importance
xgb_importance = np.mean([model.feature_importances_ for model in models['xgboost']], axis=0)
feature_importance['xgboost'] = xgb_importance

# LightGBM importance
lgb_importance = np.mean([model.feature_importances_ for model in models['lightgbm']], axis=0)
feature_importance['lightgbm'] = lgb_importance

# CatBoost importance
cb_importance = np.mean([model.get_feature_importance() for model in models['catboost']], axis=0)
feature_importance['catboost'] = cb_importance

# Random Forest importance
rf_importance = np.mean([model.feature_importances_ for model in models['random_forest']], axis=0)
feature_importance['random_forest'] = rf_importance

# Average importance across all models
feature_importance['average'] = feature_importance[['xgboost', 'lightgbm', 'catboost', 'random_forest']].mean(axis=1)
#feature_importance['average'] = feature_importance[['xgboost', 'lightgbm', 'catboost']].mean(axis=1)

# Sort by average importance and display top 20
feature_importance_sorted = feature_importance.sort_values('average', ascending=False)

print("\nTop 20 Most Important Features (Average across all models):")
print("="*60)
print(feature_importance_sorted[['feature', 'average']].head(20).to_string(index=False))

# Plot top 15 features
plt.figure(figsize=(10, 8))
top_features = feature_importance_sorted.head(15)
plt.barh(range(len(top_features)), top_features['average'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Average Feature Importance')
plt.title('Top 15 Most Important Features (Averaged Across All Models)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Select the best ensemble strategy for final submission
if best_strategy == 'Simple':
    final_predictions = ensemble_test_simple
elif best_strategy == 'Weighted':
    final_predictions = ensemble_test_weighted
elif best_strategy == 'Rank':
    final_predictions = ensemble_test_rank
else:
    final_predictions = ensemble_test_optimized

# Ensure predictions are within [0, 1] range
final_predictions = np.clip(final_predictions, 0, 1)

# Create submission dataframe
submission_df = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': final_predictions
})

# Save the best ensemble submission
#submission_df.to_csv(f'submission_ensemble_{best_strategy.lower()}.csv', index=False)
submission_df.to_csv(f'submission.csv', index=False)
print(f"\nSubmission file created: submission.csv")
print(f"Using {best_strategy} ensemble strategy with CV AUC: {best_ensemble_scores[best_strategy]:.5f}")

# Also save individual model predictions for comparison
for model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
    individual_submission = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': np.clip(test_predictions[model_name], 0, 1)
    })
    individual_submission.to_csv(f'submission_{model_name}.csv', index=False)
    print(f"Individual submission saved: submission_{model_name}.csv")

# Display sample of final submission
print("\nSample of final submission:")
print(submission_df.head(10))
print(f"\nPrediction statistics:")
print(submission_df['loan_paid_back'].describe())


# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Model performance comparison
ax1 = axes[0, 0]
model_names = list(auc_scores.keys()) + ['Simple\nEnsemble', 'Weighted\nEnsemble', 'Rank\nEnsemble', 'Optimized\nEnsemble']
model_scores = [np.mean(auc_scores[m]) for m in auc_scores.keys()] + [ensemble_auc_simple, ensemble_auc_weighted, ensemble_auc_rank, ensemble_auc_optimized]
colors = ['skyblue'] * 3 + ['lightcoral'] * 4
bars = ax1.bar(model_names, model_scores, color=colors)
ax1.set_ylabel('AUC Score')
ax1.set_title('Model Performance Comparison')
ax1.set_ylim([min(model_scores) * 0.98, max(model_scores) * 1.005])
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, score in zip(bars, model_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.5f}', ha='center', va='bottom', fontsize=8)

# Plot 2: Cross-validation scores distribution
ax2 = axes[0, 1]
cv_data = [auc_scores[m] for m in ['xgboost', 'lightgbm', 'catboost', 'random_forest']]
bp = ax2.boxplot(cv_data, labels=['XGBoost', 'LightGBM', 'CatBoost', 'RF'])
ax2.set_ylabel('AUC Score')
ax2.set_title('Cross-Validation Score Distribution')
ax2.grid(True, alpha=0.3)

# Plot 3: Prediction distribution
ax3 = axes[1, 0]
ax3.hist(final_predictions, bins=50, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Predicted Probability')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Final Test Predictions')
ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='0.5 threshold')
ax3.legend()

fig.delaxes(axes[1, 1])
plt.tight_layout()
plt.show()




