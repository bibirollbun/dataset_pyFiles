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

from catboost import CatBoostClassifier, Pool


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


df_test


print("Training Data Head:")
display(df_train.head())

print("\nTraining Data Info:")
print(df_train.info())

print("\nMissing Values in Train Data:")
print(df_train.isnull().sum())

print("\nMissing Values in Test Data:")
print(df_test.isnull().sum())

print('\nDescriptive statistics for numerical columns') 
display(df_train.describe())


# Distribution of the target variable 'accident_risk'
plt.figure(figsize=(10, 6))
sns.countplot(x='loan_paid_back', data=df_train, palette='pastel', edgecolor='black')
plt.title('Distribution of Loan Payback')
plt.xlabel('Loan Payback')
plt.ylabel('Count')
plt.show()


categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


# compact view of categorical features vs the target
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


target_col = 'loan_paid_back'

numerical_features = df_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'loan_paid_back']]
print(numerical_features)

categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)


# I will use combine_df only where I am sure I will not have data leak between df_train and df_test 


combined_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)


def create_minimal_features(df):
    """Keep only grade_rank - the ONE engineered feature that helps"""
    
    # Only this one - it has 2.92 importance, decent contribution
    df['grade_letter'] = df['grade_subgrade'].str[0]
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_rank'] = df['grade_letter'].map(grade_map)
    
    return df


combined_df = create_minimal_features(combined_df)




combined_df


#columns_to_remove = ['grade_subgrade','grade_letter']
columns_to_remove = ['grade_letter']
#combined_df.drop(columns=columns_to_remove, inplace = True)

df_train = combined_df.iloc[:len(df_train)]
df_test = combined_df.iloc[len(df_train):].drop(columns=[target_col]).reset_index(drop=True)


df_test


df_train


df_train.describe()




# def bin_numeric_features(df, n_bins=4, exclude_cols=None):
#     """
#     Automatically create quantile-based bins for all numeric features, 
#     excluding specified columns (like IDs or targets).

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input dataframe.
#     n_bins : int, optional (default=4)
#         Number of quantile bins to create (4 = quartiles).
#     exclude_cols : list, optional
#         Columns to exclude from binning.

#     Returns
#     -------
#     df_binned : pd.DataFrame
#         Copy of df with added *_bin columns for each numeric feature.
#     """
#     df_binned = df.copy()
    
#     if exclude_cols is None:
#         exclude_cols = []

#     numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
#     numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

#     for col in numeric_cols:
#         try:
#             df_binned[f'{col}_bin'] = pd.qcut(df[col], q=n_bins, labels=[f'Q{i+1}' for i in range(n_bins)])
#         except ValueError:
#             # Handle constant or low-unique columns safely
#             df_binned[f'{col}_bin'] = pd.cut(df[col], bins=n_bins, labels=[f'Q{i+1}' for i in range(n_bins)], duplicates='drop')

#     return df_binned



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



# # ============================================================================
# # Optional: Create strategic interactions with binned features
# # ============================================================================
# def create_binned_interactions(df):
#     """
#     Create interactions between binned numerics and key categoricals
#     Only the most important ones based on domain knowledge
#     """
    
#     # Credit quality × Employment (unemployed + bad credit = high risk)
#     df['credit_emp'] = (df['credit_score_bin'].astype(str) + '_' + 
#                         df['employment_status'].astype(str))
    
#     # DTI × Grade (high debt + low grade = very high risk)
#     df['debt_to_income_ratio_grade'] = (df['debt_to_income_ratio_bin'].astype(str) + '_' + 
#                        df['grade_subgrade'].astype(str))
    
#     # Interest rate × Employment (high rate + unemployed = default likely)
#     df['interest_rate_emp'] = (df['interest_rate_bin'].astype(str) + '_' + 
#                           df['employment_status'].astype(str))
    
#     # Loan size × Income (large loan + low income = can't afford)
#     df['loan_income_cross'] = (df['loan_amount_bin'].astype(str) + '_' + 
#                                df['annual_income_bin'].astype(str))
    
#     return df




# # ============================================================================
# # Usage in your pipeline
# # ============================================================================
# combined_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# # Basic feature engineering
# #combined_df['grade_letter'] = combined_df['grade_subgrade'].str[0]
# #grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
# #combined_df['grade_rank'] = combined_df['grade_letter'].map(grade_map)

# # Add binned features
# combined_df = bin_numeric_features(combined_df, exclude_cols=['id', 'loan_paid_back'])


# # Optionally add interactions (test this separately!)
# combined_df = create_binned_interactions(combined_df)



# # Drop original grade columns
# #combined_df.drop(columns=['grade_subgrade', 'grade_letter'], inplace=True)

# # Continue with your split...


# combined_df


# columns_to_remove = ['gender','marital_status','income_bin', 'interest_bin','loan_amount_bin']
# combined_df.drop(columns=columns_to_remove, inplace = True)



# df_train = combined_df.iloc[:len(df_train)]
# df_test = combined_df.iloc[len(df_train):].drop(columns=[target_col]).reset_index(drop=True)


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, cross_val_predict
# from sklearn.metrics import roc_auc_score
# from sklearn.linear_model import LogisticRegression, RidgeClassifier
# from sklearn.preprocessing import LabelEncoder
# from scipy.stats import rankdata
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# import optuna
# import warnings
# warnings.filterwarnings('ignore')

# # =============================================================================
# # COMPREHENSIVE ENSEMBLE WITH ALL METHODS
# # =============================================================================

# class ComprehensiveEnsemble:
#     """
#     Complete ensemble implementation with:
#     - Weighted averaging
#     - Stacking with meta-features
#     - Multi-level stacking
#     - Rank averaging
#     - Bayesian averaging
#     - Hill climbing optimization
#     """
    
#     def __init__(self, n_splits=5, random_state=42):
#         self.n_splits = n_splits
#         self.random_state = random_state
#         self.base_models = {}
#         self.meta_models = {}
#         self.ensemble_results = {}
        
#         # Store predictions
#         self.oof_predictions = {}
#         self.test_predictions = {}
        
#     def prepare_data(self, df_train, df_test, target_col='loan_paid_back'):
#         """Prepare training and test data"""
#         self.X = df_train.drop(['id', target_col], axis=1, errors='ignore')
#         self.y = df_train[target_col]
#         self.X_test = df_test.drop(['id'], axis=1, errors='ignore')
#         self.test_ids = df_test['id'] if 'id' in df_test.columns else None
        
#         # Identify feature types
#         self.categorical_features = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
#         self.numerical_features = self.X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        
#         print(f"Dataset shape: {self.X.shape}")
#         print(f"Categorical features: {len(self.categorical_features)}")
#         print(f"Numerical features: {len(self.numerical_features)}")
        
#     def get_model_params(self):
#         """Your proven model parameters"""
        
#         # YOUR CatBoost parameters (proven to work at 0.9236!)
#         catboost_params = {
#             'iterations': 2000,
#             'learning_rate': 0.05,
#             'depth': 5,
#             'l2_leaf_reg': 3,
#             'loss_function': 'Logloss',
#             'eval_metric': 'AUC',
#             'random_seed': self.random_state,
#             'od_type': 'Iter',
#             'od_wait': 100,
#             'grow_policy': 'Lossguide',
#             'thread_count': -1,
#             'verbose': 100,
#             'allow_writing_files': False
#         }
        
#         # LightGBM parameters
#         lgb_params = {
#             'objective': 'binary',
#             'metric': 'auc',
#             'boosting_type': 'gbdt',
#             'n_estimators': 2000,
#             'max_depth': 5,
#             'num_leaves': 31,
#             'learning_rate': 0.05,
#             'colsample_bytree': 0.8,
#             'subsample': 0.8,
#             'subsample_freq': 1,
#             'min_child_samples': 20,
#             'reg_alpha': 0.05,
#             'reg_lambda': 0.1,
#             'random_state': self.random_state,
#             'n_jobs': -1,
#             'verbose': -1
#         }
        
#         # XGBoost parameters
#         xgb_params = {
#             'objective': 'binary:logistic',
#             'eval_metric': 'auc',
#             'n_estimators': 2000,
#             'max_depth': 5,
#             'learning_rate': 0.05,
#             'colsample_bytree': 0.8,
#             'subsample': 0.8,
#             'min_child_weight': 20,
#             'reg_alpha': 0.05,
#             'reg_lambda': 0.1,
#             'gamma': 0.01,
#             'random_state': self.random_state,
#             'n_jobs': -1,
#             'tree_method': 'hist',
#             'verbosity': 0
#         }
        
#         return catboost_params, lgb_params, xgb_params
    
#     def train_base_models(self):
#         """Train CatBoost, LightGBM, and XGBoost with cross-validation"""
#         print("\n" + "="*70)
#         print("TRAINING BASE MODELS")
#         print("="*70)
        
#         catboost_params, lgb_params, xgb_params = self.get_model_params()
        
#         skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
#         # Initialize OOF and test predictions
#         oof_cat = np.zeros(len(self.X))
#         oof_lgb = np.zeros(len(self.X))
#         oof_xgb = np.zeros(len(self.X))
        
#         test_cat = np.zeros(len(self.X_test))
#         test_lgb = np.zeros(len(self.X_test))
#         test_xgb = np.zeros(len(self.X_test))
        
#         cat_scores, lgb_scores, xgb_scores = [], [], []
        
#         for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y), 1):
#             print(f"\n{'='*60}")
#             print(f"FOLD {fold}/{self.n_splits}")
#             print(f"{'='*60}")
            
#             X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
#             y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
#             # 1. CATBOOST
#             print("Training CatBoost...")
#             cat_model = CatBoostClassifier(**catboost_params)
#             cat_model.fit(
#                 X_train, y_train,
#                 eval_set=(X_val, y_val),
#                 cat_features=self.categorical_features
#             )
            
#             oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
#             test_cat += cat_model.predict_proba(self.X_test)[:, 1] / self.n_splits
            
#             cat_score = roc_auc_score(y_val, oof_cat[val_idx])
#             cat_scores.append(cat_score)
#             print(f"  CatBoost  Fold {fold} AUC: {cat_score:.5f}")
            
#             # 2. LIGHTGBM
#             print("Training LightGBM...")
            
#             # Encode categoricals for LightGBM
#             X_train_lgb = X_train.copy()
#             X_val_lgb = X_val.copy()
#             X_test_lgb = self.X_test.copy()
            
#             label_encoders = {}
#             for col in self.categorical_features:
#                 le = LabelEncoder()
#                 all_values = pd.concat([X_train[col], X_val[col]]).astype(str).fillna('missing')
#                 le.fit(all_values)
                
#                 X_train_lgb[col] = le.transform(X_train[col].astype(str).fillna('missing'))
#                 X_val_lgb[col] = le.transform(X_val[col].astype(str).fillna('missing'))
                
#                 test_values = self.X_test[col].astype(str).fillna('missing')
#                 test_values = test_values.map(lambda x: x if x in le.classes_ else 'missing')
#                 X_test_lgb[col] = le.transform(test_values)
                
#                 label_encoders[col] = le
            
#             lgb_model = lgb.LGBMClassifier(**lgb_params)
#             lgb_model.fit(
#                 X_train_lgb, y_train,
#                 eval_set=[(X_val_lgb, y_val)],
#                 categorical_feature=self.categorical_features,
#                 callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
#             )
            
#             oof_lgb[val_idx] = lgb_model.predict_proba(X_val_lgb)[:, 1]
#             test_lgb += lgb_model.predict_proba(X_test_lgb)[:, 1] / self.n_splits
            
#             lgb_score = roc_auc_score(y_val, oof_lgb[val_idx])
#             lgb_scores.append(lgb_score)
#             print(f"  LightGBM  Fold {fold} AUC: {lgb_score:.5f}")
            
#             # 3. XGBOOST
#             print("Training XGBoost...")
            
#             xgb_model = xgb.XGBClassifier(**xgb_params)
#             xgb_model.fit(
#                 X_train_lgb, y_train,
#                 eval_set=[(X_val_lgb, y_val)],
#                 verbose=False,
#                 early_stopping_rounds=100
#             )
            
#             oof_xgb[val_idx] = xgb_model.predict_proba(X_val_lgb)[:, 1]
#             test_xgb += xgb_model.predict_proba(X_test_lgb)[:, 1] / self.n_splits
            
#             xgb_score = roc_auc_score(y_val, oof_xgb[val_idx])
#             xgb_scores.append(xgb_score)
#             print(f"  XGBoost   Fold {fold} AUC: {xgb_score:.5f}")
        
#         # Store predictions
#         self.oof_predictions['catboost'] = oof_cat
#         self.oof_predictions['lightgbm'] = oof_lgb
#         self.oof_predictions['xgboost'] = oof_xgb
        
#         self.test_predictions['catboost'] = test_cat
#         self.test_predictions['lightgbm'] = test_lgb
#         self.test_predictions['xgboost'] = test_xgb
        
#         # Calculate CV scores
#         cat_cv = roc_auc_score(self.y, oof_cat)
#         lgb_cv = roc_auc_score(self.y, oof_lgb)
#         xgb_cv = roc_auc_score(self.y, oof_xgb)
        
#         print("\n" + "="*70)
#         print("BASE MODEL PERFORMANCE")
#         print("="*70)
#         print(f"CatBoost  CV AUC: {cat_cv:.5f} ± {np.std(cat_scores):.5f}")
#         print(f"LightGBM  CV AUC: {lgb_cv:.5f} ± {np.std(lgb_scores):.5f}")
#         print(f"XGBoost   CV AUC: {xgb_cv:.5f} ± {np.std(xgb_scores):.5f}")
        
#         self.base_scores = {'catboost': cat_cv, 'lightgbm': lgb_cv, 'xgboost': xgb_cv}
        
#     def create_meta_features(self, cat_pred, lgb_pred, xgb_pred):
#         """Create advanced meta-features for stacking"""
#         features = np.column_stack([
#             # Base predictions
#             cat_pred, 
#             lgb_pred, 
#             xgb_pred,
            
#             # Interaction features
#             cat_pred * lgb_pred,
#             cat_pred * xgb_pred,
#             lgb_pred * xgb_pred,
            
#             # Triple interaction
#             cat_pred * lgb_pred * xgb_pred,
            
#             # Squared features
#             cat_pred ** 2,
#             lgb_pred ** 2,
#             xgb_pred ** 2,
            
#             # Disagreement features
#             np.abs(cat_pred - lgb_pred),
#             np.abs(cat_pred - xgb_pred),
#             np.abs(lgb_pred - xgb_pred),
            
#             # Statistical features
#             np.mean([cat_pred, lgb_pred, xgb_pred], axis=0),
#             np.std([cat_pred, lgb_pred, xgb_pred], axis=0),
#             np.max([cat_pred, lgb_pred, xgb_pred], axis=0),
#             np.min([cat_pred, lgb_pred, xgb_pred], axis=0),
#             np.median([cat_pred, lgb_pred, xgb_pred], axis=0),
            
#             # Range and spread
#             np.max([cat_pred, lgb_pred, xgb_pred], axis=0) - 
#             np.min([cat_pred, lgb_pred, xgb_pred], axis=0),
            
#             # Confidence features
#             np.where(cat_pred > 0.5, cat_pred, 1 - cat_pred),
#             np.where(lgb_pred > 0.5, lgb_pred, 1 - lgb_pred),
#             np.where(xgb_pred > 0.5, xgb_pred, 1 - xgb_pred),
#         ])
        
#         return features
    
#     def weighted_average_ensemble(self):
#         """Find optimal weights using grid search"""
#         print("\n" + "="*70)
#         print("WEIGHTED AVERAGE ENSEMBLE")
#         print("="*70)
        
#         best_score = 0
#         best_weights = None
        
#         oof_cat = self.oof_predictions['catboost']
#         oof_lgb = self.oof_predictions['lightgbm']
#         oof_xgb = self.oof_predictions['xgboost']
        
#         # Grid search for weights
#         for w1 in np.arange(0, 1.05, 0.05):
#             for w2 in np.arange(0, 1.05 - w1, 0.05):
#                 w3 = round(1 - w1 - w2, 2)
#                 if w3 < 0 or w3 > 1:
#                     continue
                
#                 oof_blend = w1 * oof_cat + w2 * oof_lgb + w3 * oof_xgb
#                 score = roc_auc_score(self.y, oof_blend)
                
#                 if score > best_score:
#                     best_score = score
#                     best_weights = (w1, w2, w3)
        
#         # Create final predictions
#         test_cat = self.test_predictions['catboost']
#         test_lgb = self.test_predictions['lightgbm']
#         test_xgb = self.test_predictions['xgboost']
        
#         w1, w2, w3 = best_weights
#         test_final = w1 * test_cat + w2 * test_lgb + w3 * test_xgb
        
#         print(f"Best weights (CAT, LGB, XGB): {best_weights}")
#         print(f"Weighted Average CV AUC: {best_score:.5f}")
        
#         self.ensemble_results['weighted_average'] = {
#             'cv_score': best_score,
#             'test_predictions': test_final,
#             'weights': best_weights
#         }
        
#     def rank_average_ensemble(self):
#         """Rank averaging ensemble"""
#         print("\n" + "="*70)
#         print("RANK AVERAGE ENSEMBLE")
#         print("="*70)
        
#         # Convert to ranks for OOF
#         oof_cat_rank = rankdata(self.oof_predictions['catboost']) / len(self.oof_predictions['catboost'])
#         oof_lgb_rank = rankdata(self.oof_predictions['lightgbm']) / len(self.oof_predictions['lightgbm'])
#         oof_xgb_rank = rankdata(self.oof_predictions['xgboost']) / len(self.oof_predictions['xgboost'])
        
#         oof_rank_avg = np.mean([oof_cat_rank, oof_lgb_rank, oof_xgb_rank], axis=0)
        
#         # Convert to ranks for test
#         test_cat_rank = rankdata(self.test_predictions['catboost']) / len(self.test_predictions['catboost'])
#         test_lgb_rank = rankdata(self.test_predictions['lightgbm']) / len(self.test_predictions['lightgbm'])
#         test_xgb_rank = rankdata(self.test_predictions['xgboost']) / len(self.test_predictions['xgboost'])
        
#         test_rank_avg = np.mean([test_cat_rank, test_lgb_rank, test_xgb_rank], axis=0)
        
#         cv_score = roc_auc_score(self.y, oof_rank_avg)
#         print(f"Rank Average CV AUC: {cv_score:.5f}")
        
#         self.ensemble_results['rank_average'] = {
#             'cv_score': cv_score,
#             'test_predictions': test_rank_avg
#         }
        
#     def bayesian_model_averaging(self):
#         """Bayesian Model Averaging"""
#         print("\n" + "="*70)
#         print("BAYESIAN MODEL AVERAGING")
#         print("="*70)
        
#         # Use base model scores for weights
#         scores = np.array([
#             self.base_scores['catboost'],
#             self.base_scores['lightgbm'],
#             self.base_scores['xgboost']
#         ])
        
#         # Convert scores to weights using softmax with temperature
#         temperature = 10
#         weights = np.exp(scores * temperature)
#         weights = weights / weights.sum()
        
#         print(f"Bayesian weights (CAT, LGB, XGB): {weights}")
        
#         # Create predictions
#         oof_bayesian = np.average([
#             self.oof_predictions['catboost'],
#             self.oof_predictions['lightgbm'],
#             self.oof_predictions['xgboost']
#         ], weights=weights, axis=0)
        
#         test_bayesian = np.average([
#             self.test_predictions['catboost'],
#             self.test_predictions['lightgbm'],
#             self.test_predictions['xgboost']
#         ], weights=weights, axis=0)
        
#         cv_score = roc_auc_score(self.y, oof_bayesian)
#         print(f"Bayesian Average CV AUC: {cv_score:.5f}")
        
#         self.ensemble_results['bayesian_average'] = {
#             'cv_score': cv_score,
#             'test_predictions': test_bayesian,
#             'weights': weights
#         }
        
#     def hill_climbing_optimization(self, n_iterations=1000):
#         """Hill climbing to find optimal weights"""
#         print("\n" + "="*70)
#         print("HILL CLIMBING OPTIMIZATION")
#         print("="*70)
        
#         predictions = [
#             self.oof_predictions['catboost'],
#             self.oof_predictions['lightgbm'],
#             self.oof_predictions['xgboost']
#         ]
        
#         n_models = 3
#         best_weights = np.ones(n_models) / n_models
#         best_score = roc_auc_score(self.y, np.average(predictions, weights=best_weights, axis=0))
        
#         for iteration in range(n_iterations):
#             # Perturb weights
#             new_weights = best_weights + np.random.randn(n_models) * 0.01
#             new_weights = np.clip(new_weights, 0, 1)
#             new_weights = new_weights / (new_weights.sum() + 1e-10)
            
#             blend = np.average(predictions, weights=new_weights, axis=0)
#             score = roc_auc_score(self.y, blend)
            
#             if score > best_score:
#                 best_score = score
#                 best_weights = new_weights
                
#             if iteration % 100 == 0:
#                 print(f"  Iteration {iteration}: Best score = {best_score:.5f}")
        
#         # Create test predictions
#         test_predictions = [
#             self.test_predictions['catboost'],
#             self.test_predictions['lightgbm'],
#             self.test_predictions['xgboost']
#         ]
#         test_hill = np.average(test_predictions, weights=best_weights, axis=0)
        
#         print(f"Hill Climb weights (CAT, LGB, XGB): {best_weights}")
#         print(f"Hill Climbing CV AUC: {best_score:.5f}")
        
#         self.ensemble_results['hill_climbing'] = {
#             'cv_score': best_score,
#             'test_predictions': test_hill,
#             'weights': best_weights
#         }
        
#     def stacking_ensemble(self):
#         """Stacking with multiple meta-learners"""
#         print("\n" + "="*70)
#         print("STACKING ENSEMBLE")
#         print("="*70)
        
#         # Create meta-features
#         meta_features_train = self.create_meta_features(
#             self.oof_predictions['catboost'],
#             self.oof_predictions['lightgbm'],
#             self.oof_predictions['xgboost']
#         )
        
#         meta_features_test = self.create_meta_features(
#             self.test_predictions['catboost'],
#             self.test_predictions['lightgbm'],
#             self.test_predictions['xgboost']
#         )
        
#         # Define meta-models
#         meta_models = {
#             'LogisticRegression': LogisticRegression(C=0.5, max_iter=1000),
#             'Ridge': RidgeClassifier(alpha=0.5),
#             'LightGBM_meta': lgb.LGBMClassifier(
#                 n_estimators=200,
#                 max_depth=3,
#                 learning_rate=0.1,
#                 colsample_bytree=0.7,
#                 subsample=0.7,
#                 random_state=self.random_state,
#                 verbose=-1
#             ),
#             'XGBoost_meta': xgb.XGBClassifier(
#                 n_estimators=200,
#                 max_depth=3,
#                 learning_rate=0.1,
#                 colsample_bytree=0.7,
#                 subsample=0.7,
#                 random_state=self.random_state,
#                 verbosity=0
#             ),
#             'CatBoost_meta': CatBoostClassifier(
#                 iterations=200,
#                 depth=3,
#                 learning_rate=0.1,
#                 random_seed=self.random_state,
#                 verbose=0
#             )
#         }
        
#         stacking_predictions = {}
#         best_stacking_score = 0
#         best_stacking_model = None
#         best_stacking_test = None
        
#         for name, model in meta_models.items():
#             print(f"\nTraining {name} meta-model...")
            
#             # Get OOF predictions for meta-model
#             if hasattr(model, 'predict_proba'):
#                 meta_oof = cross_val_predict(
#                     model, meta_features_train, self.y, 
#                     cv=5, method='predict_proba', n_jobs=-1
#                 )[:, 1]
#             else:
#                 # For Ridge
#                 meta_oof = cross_val_predict(
#                     model, meta_features_train, self.y, 
#                     cv=5, method='decision_function', n_jobs=-1
#                 )
#                 # Convert to probabilities
#                 meta_oof = 1 / (1 + np.exp(-meta_oof))
            
#             score = roc_auc_score(self.y, meta_oof)
#             print(f"  {name} Stacking CV AUC: {score:.5f}")
            
#             # Train on full data for test predictions
#             model.fit(meta_features_train, self.y)
            
#             if hasattr(model, 'predict_proba'):
#                 test_preds = model.predict_proba(meta_features_test)[:, 1]
#             else:
#                 test_preds = model.decision_function(meta_features_test)
#                 test_preds = 1 / (1 + np.exp(-test_preds))
            
#             stacking_predictions[name] = {
#                 'cv_score': score,
#                 'test_predictions': test_preds
#             }
            
#             if score > best_stacking_score:
#                 best_stacking_score = score
#                 best_stacking_model = name
#                 best_stacking_test = test_preds
        
#         print(f"\nBest Stacking Model: {best_stacking_model}")
#         print(f"Best Stacking CV AUC: {best_stacking_score:.5f}")
        
#         self.ensemble_results['stacking'] = {
#             'cv_score': best_stacking_score,
#             'test_predictions': best_stacking_test,
#             'best_model': best_stacking_model,
#             'all_models': stacking_predictions
#         }
        
#     def multi_level_stacking(self):
#         """Multi-level stacking ensemble"""
#         print("\n" + "="*70)
#         print("MULTI-LEVEL STACKING")
#         print("="*70)
        
#         # Level 1: Base models (already have these)
#         level1_train = np.column_stack([
#             self.oof_predictions['catboost'],
#             self.oof_predictions['lightgbm'],
#             self.oof_predictions['xgboost']
#         ])
        
#         level1_test = np.column_stack([
#             self.test_predictions['catboost'],
#             self.test_predictions['lightgbm'],
#             self.test_predictions['xgboost']
#         ])
        
#         # Level 2: Train new models on Level 1 predictions
#         print("\nTraining Level 2 models...")
        
#         level2_models = {
#             'lgb_l2': lgb.LGBMClassifier(
#                 n_estimators=150, max_depth=3, learning_rate=0.1,
#                 random_state=self.random_state, verbose=-1
#             ),
#             'xgb_l2': xgb.XGBClassifier(
#                 n_estimators=150, max_depth=3, learning_rate=0.1,
#                 random_state=self.random_state, verbosity=0
#             ),
#             'lr_l2': LogisticRegression(C=1.0, max_iter=1000)
#         }
        
#         level2_train = []
#         level2_test = []
        
#         for name, model in level2_models.items():
#             # Get OOF predictions
#             oof_pred = cross_val_predict(
#                 model, level1_train, self.y, 
#                 cv=5, method='predict_proba', n_jobs=-1
#             )[:, 1]
            
#             level2_train.append(oof_pred)
            
#             # Train on full data and predict on test
#             model.fit(level1_train, self.y)
#             test_pred = model.predict_proba(level1_test)[:, 1]
#             level2_test.append(test_pred)
            
#             score = roc_auc_score(self.y, oof_pred)
#             print(f"  {name} Level 2 AUC: {score:.5f}")
        
#         level2_train = np.column_stack(level2_train)
#         level2_test = np.column_stack(level2_test)
        
#         # Level 3: Final meta-model
#         print("\nTraining Level 3 (final) model...")
        
#         final_model = LogisticRegression(C=0.5, max_iter=1000)
        
#         # Get final OOF predictions
#         final_oof = cross_val_predict(
#             final_model, level2_train, self.y,
#             cv=5, method='predict_proba', n_jobs=-1
#         )[:, 1]
        
#         # Train on full data for test predictions
#         final_model.fit(level2_train, self.y)
#         final_test = final_model.predict_proba(level2_test)[:, 1]
        
#         cv_score = roc_auc_score(self.y, final_oof)
#         print(f"\nMulti-Level Stacking CV AUC: {cv_score:.5f}")
        
#         self.ensemble_results['multi_level_stacking'] = {
#             'cv_score': cv_score,
#             'test_predictions': final_test
#         }
        
#     def optimize_stacking_with_optuna(self, n_trials=50):
#         """Use Optuna to optimize stacking meta-model"""
#         print("\n" + "="*70)
#         print("OPTUNA OPTIMIZATION FOR STACKING")
#         print("="*70)
        
#         # Create meta-features
#         meta_features_train = self.create_meta_features(
#             self.oof_predictions['catboost'],
#             self.oof_predictions['lightgbm'],
#             self.oof_predictions['xgboost']
#         )
        
#         meta_features_test = self.create_meta_features(
#             self.test_predictions['catboost'],
#             self.test_predictions['lightgbm'],
#             self.test_predictions['xgboost']
#         )
        
#         def objective(trial):
#             # Choose meta-model type
#             model_type = trial.suggest_categorical('model_type', ['lgb', 'xgb', 'catboost'])
            
#             if model_type == 'lgb':
#                 params = {
#                     'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#                     'max_depth': trial.suggest_int('max_depth', 2, 5),
#                     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#                     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
#                     'subsample': trial.suggest_float('subsample', 0.3, 1.0),
#                     'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
#                     'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
#                     'random_state': self.random_state,
#                     'verbose': -1
#                 }
#                 model = lgb.LGBMClassifier(**params)
                
#             elif model_type == 'xgb':
#                 params = {
#                     'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#                     'max_depth': trial.suggest_int('max_depth', 2, 5),
#                     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#                     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
#                     'subsample': trial.suggest_float('subsample', 0.3, 1.0),
#                     'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
#                     'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
#                     'random_state': self.random_state,
#                     'verbosity': 0
#                 }
#                 model = xgb.XGBClassifier(**params)
                
#             else:  # catboost
#                 params = {
#                     'iterations': trial.suggest_int('iterations', 50, 300),
#                     'depth': trial.suggest_int('depth', 2, 5),
#                     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#                     'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#                     'random_seed': self.random_state,
#                     'verbose': 0
#                 }
#                 model = CatBoostClassifier(**params)
            
#             # Cross-validation score
#             oof_pred = cross_val_predict(
#                 model, meta_features_train, self.y,
#                 cv=5, method='predict_proba', n_jobs=-1
#             )[:, 1]
            
#             score = roc_auc_score(self.y, oof_pred)
#             return score
        
#         # Run optimization
#         study = optuna.create_study(direction='maximize', study_name='stacking_optimization')
#         study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
#         print(f"\nBest Optuna score: {study.best_value:.5f}")
#         print(f"Best params: {study.best_params}")
        
#         # Train best model
#         best_params = study.best_params
#         model_type = best_params.pop('model_type')
        
#         if model_type == 'lgb':
#             best_model = lgb.LGBMClassifier(**best_params, random_state=self.random_state, verbose=-1)
#         elif model_type == 'xgb':
#             best_model = xgb.XGBClassifier(**best_params, random_state=self.random_state, verbosity=0)
#         else:
#             best_model = CatBoostClassifier(**best_params, random_seed=self.random_state, verbose=0)
        
#         # Get final predictions
#         best_model.fit(meta_features_train, self.y)
#         test_optuna = best_model.predict_proba(meta_features_test)[:, 1]
        
#         self.ensemble_results['optuna_stacking'] = {
#             'cv_score': study.best_value,
#             'test_predictions': test_optuna,
#             'best_params': study.best_params
#         }
        
#     def blend_best_ensembles(self):
#         """Blend the best ensemble methods"""
#         print("\n" + "="*70)
#         print("FINAL ENSEMBLE BLEND")
#         print("="*70)
        
#         # Select top ensemble methods
#         ensemble_scores = [(name, res['cv_score']) for name, res in self.ensemble_results.items()]
#         ensemble_scores.sort(key=lambda x: x[1], reverse=True)
        
#         print("\nAll ensemble scores:")
#         for name, score in ensemble_scores:
#             print(f"  {name}: {score:.5f}")
        
#         # Blend top 3 methods
#         top_methods = ensemble_scores[:3]
#         print(f"\nBlending top 3 methods: {[m[0] for m in top_methods]}")
        
#         # Weight by performance
#         weights = np.array([score for _, score in top_methods])
#         weights = np.exp(weights * 20)  # Sharpen differences
#         weights = weights / weights.sum()
        
#         # Create final blend
#         final_test = np.zeros(len(self.X_test))
#         for i, (method_name, _) in enumerate(top_methods):
#             final_test += weights[i] * self.ensemble_results[method_name]['test_predictions']
        
#         print(f"Blend weights: {weights}")
        
#         # Also create simple average of top 3
#         simple_avg_test = np.mean([
#             self.ensemble_results[name]['test_predictions'] 
#             for name, _ in top_methods
#         ], axis=0)
        
#         return final_test, simple_avg_test
    
#     def run_all_ensembles(self):
#         """Run all ensemble methods"""
        
#         # 1. Train base models
#         self.train_base_models()
        
#         # 2. Run all ensemble methods
#         self.weighted_average_ensemble()
#         self.rank_average_ensemble()
#         self.bayesian_model_averaging()
#         self.hill_climbing_optimization(n_iterations=500)
#         self.stacking_ensemble()
#         self.multi_level_stacking()
#         self.optimize_stacking_with_optuna(n_trials=30)
        
#         # 3. Create final blend
#         final_blend, simple_avg = self.blend_best_ensembles()
        
#         # 4. Save best submission
#         best_method = max(self.ensemble_results.items(), key=lambda x: x[1]['cv_score'])
#         print("\n" + "="*70)
#         print(f"BEST SINGLE METHOD: {best_method[0]} with CV: {best_method[1]['cv_score']:.5f}")
#         print("="*70)
        
#         # Save multiple submissions
#         submissions = {
#             'best_single': best_method[1]['test_predictions'],
#             'final_blend': final_blend,
#             'top3_simple_avg': simple_avg
#         }
        
#         for name, predictions in submissions.items():
#             submission = pd.DataFrame({
#                 'id': self.test_ids,
#                 'loan_paid_back': predictions
#             })
#             filename = f'submission_{name}.csv'
#             submission.to_csv(filename, index=False)
#             print(f"✅ Saved: {filename}")
        
#         # Also save main submission
#         submission = pd.DataFrame({
#             'id': self.test_ids,
#             'loan_paid_back': final_blend  # Use the weighted blend
#         })
#         submission.to_csv('submission.csv', index=False)
#         print(f"✅ Saved: submission.csv (main submission)")
        
#         return self.ensemble_results

# # =============================================================================
# # MAIN EXECUTION
# # =============================================================================

# def run_comprehensive_ensemble(df_train, df_test, target_col='loan_paid_back'):
#     """Main function to run all ensembles"""
    
#     print("="*70)
#     print("COMPREHENSIVE ENSEMBLE SYSTEM")
#     print("="*70)
#     print("Methods included:")
#     print("  1. Weighted Average Optimization")
#     print("  2. Rank Averaging")
#     print("  3. Bayesian Model Averaging")
#     print("  4. Hill Climbing Optimization")
#     print("  5. Stacking with Meta-features")
#     print("  6. Multi-level Stacking")
#     print("  7. Optuna-optimized Stacking")
#     print("  8. Final Ensemble Blend")
#     print("="*70)
    
#     # Initialize and run ensemble
#     ensemble = ComprehensiveEnsemble(n_splits=5, random_state=42)
#     ensemble.prepare_data(df_train, df_test, target_col)
#     results = ensemble.run_all_ensembles()
    
#     print("\n" + "="*70)
#     print("FINAL SUMMARY")
#     print("="*70)
#     print("Expected outcomes:")
#     print("• Base CatBoost: ~0.9236")
#     print("• Basic ensemble: ~0.924")
#     print("• Advanced stacking: ~0.925-0.926")
#     print("• Multi-level + optimization: ~0.926-0.927")
    
#     return results

# # Execute the ensemble
# # Note: Make sure df_train and df_test are loaded before running this
# results = run_comprehensive_ensemble(df_train, df_test)


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostClassifier
# import warnings
# warnings.filterwarnings('ignore')

# # =============================================================================
# # OPTIMIZED ENSEMBLE WITH YOUR CATBOOST PARAMS
# # =============================================================================

# def run_optimized_ensemble(df_train, df_test, target_col='loan_paid_back'):
#     """
#     Ensemble with your proven CatBoost params
#     """
    
#     print("="*70)
#     print("OPTIMIZED ENSEMBLE - NO FEATURE ENGINEERING")
#     print("="*70)
    
#     # Prepare data
#     X = df_train.drop(['id', target_col], axis=1, errors='ignore')
#     y = df_train[target_col]
#     X_test = df_test.drop(['id'], axis=1, errors='ignore')
#     test_ids = df_test['id'] if 'id' in df_test.columns else None
    
#     # Identify categorical columns
#     categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
#     numerical_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
#     print(f"\nDataset shape: {X.shape}")
#     print(f"Categorical features: {len(categorical_features)}")
#     print(f"Numerical features: {len(numerical_features)}")
    
#     # ==========================================================================
#     # OPTIMIZED PARAMETERS
#     # ==========================================================================
    
#     # YOUR CatBoost parameters (proven to work!)
#     catboost_params = {
#         'iterations': 2000,
#         'learning_rate': 0.05,
#         'depth': 5,
#         'l2_leaf_reg': 3,
#         'loss_function': 'Logloss',
#         'eval_metric': 'AUC',
#         'random_seed': 42,
#         'od_type': 'Iter',
#         'od_wait': 100,
#         'grow_policy': 'Lossguide',
#         'thread_count': -1,
#         'verbose': 100,
#         'allow_writing_files': False
#     }
    
#     # LightGBM parameters (adjusted to match CatBoost style)
#     lgb_params = {
#         'objective': 'binary',
#         'metric': 'auc',
#         'boosting_type': 'gbdt',
#         'n_estimators': 2000,
#         'max_depth': 5,  # Match CatBoost depth
#         'num_leaves': 31,  # 2^5 - 1
#         'learning_rate': 0.05,  # Match CatBoost LR
#         'colsample_bytree': 0.8,
#         'subsample': 0.8,
#         'subsample_freq': 1,
#         'min_child_samples': 20,
#         'reg_alpha': 0.05,
#         'reg_lambda': 0.1,
#         'random_state': 42,
#         'n_jobs': -1,
#         'verbose': -1
#     }
    
#     # XGBoost parameters (adjusted to match)
#     xgb_params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'n_estimators': 2000,
#         'max_depth': 5,  # Match CatBoost depth
#         'learning_rate': 0.05,  # Match CatBoost LR
#         'colsample_bytree': 0.8,
#         'subsample': 0.8,
#         'min_child_weight': 20,
#         'reg_alpha': 0.05,
#         'reg_lambda': 0.1,
#         'gamma': 0.01,
#         'random_state': 42,
#         'n_jobs': -1,
#         'tree_method': 'hist',
#         'verbosity': 0
#     }
    
#     # ==========================================================================
#     # 5-FOLD CROSS-VALIDATION (matching your setup)
#     # ==========================================================================
    
#     N_SPLITS = 5  # Match your 5-fold setup
#     skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
#     # Store OOF predictions
#     oof_lgb = np.zeros(len(X))
#     oof_xgb = np.zeros(len(X))
#     oof_cat = np.zeros(len(X))
    
#     # Store test predictions
#     test_lgb = np.zeros(len(X_test))
#     test_xgb = np.zeros(len(X_test))
#     test_cat = np.zeros(len(X_test))
    
#     # Track scores
#     lgb_scores = []
#     xgb_scores = []
#     cat_scores = []
    
#     print(f"\nRunning {N_SPLITS}-fold cross-validation...")
#     print("-" * 70)
    
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#         print(f"\n{'='*60}")
#         print(f"FOLD {fold}/{N_SPLITS}")
#         print(f"{'='*60}")
        
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
#         # ======================================================================
#         # 1. CATBOOST (YOUR CONFIGURATION)
#         # ======================================================================
#         print("Training CatBoost...")
        
#         cat_model = CatBoostClassifier(**catboost_params)
#         cat_model.fit(
#             X_train, y_train,
#             eval_set=(X_val, y_val),
#             cat_features=categorical_features  # Your way of passing categoricals
#         )
        
#         oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
#         test_cat += cat_model.predict_proba(X_test)[:, 1] / N_SPLITS
        
#         cat_score = roc_auc_score(y_val, oof_cat[val_idx])
#         cat_scores.append(cat_score)
#         print(f"  CatBoost  Fold {fold} AUC: {cat_score:.5f}")
        
#         # ======================================================================
#         # 2. LIGHTGBM
#         # ======================================================================
#         print("Training LightGBM...")
        
#         # LabelEncode categoricals for LightGBM
#         from sklearn.preprocessing import LabelEncoder
#         X_train_lgb = X_train.copy()
#         X_val_lgb = X_val.copy()
#         X_test_lgb = X_test.copy()
        
#         for col in categorical_features:
#             le = LabelEncoder()
#             # Fit on train+val
#             all_values = pd.concat([X_train[col], X_val[col]]).astype(str).fillna('missing')
#             le.fit(all_values)
            
#             # Transform
#             X_train_lgb[col] = le.transform(X_train[col].astype(str).fillna('missing'))
#             X_val_lgb[col] = le.transform(X_val[col].astype(str).fillna('missing'))
            
#             # Handle test set
#             test_values = X_test[col].astype(str).fillna('missing')
#             test_values = test_values.map(lambda x: x if x in le.classes_ else 'missing')
#             X_test_lgb[col] = le.transform(test_values)
        
#         lgb_model = lgb.LGBMClassifier(**lgb_params)
#         lgb_model.fit(
#             X_train_lgb, y_train,
#             eval_set=[(X_val_lgb, y_val)],
#             categorical_feature=categorical_features,
#             callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
#         )
        
#         oof_lgb[val_idx] = lgb_model.predict_proba(X_val_lgb)[:, 1]
#         test_lgb += lgb_model.predict_proba(X_test_lgb)[:, 1] / N_SPLITS
        
#         lgb_score = roc_auc_score(y_val, oof_lgb[val_idx])
#         lgb_scores.append(lgb_score)
#         print(f"  LightGBM  Fold {fold} AUC: {lgb_score:.5f}")
        
#         # ======================================================================
#         # 3. XGBOOST
#         # ======================================================================
#         print("Training XGBoost...")
        
#         # Prepare data for XGBoost (similar to LightGBM)
#         X_train_xgb = X_train_lgb.copy()  # Use already encoded data
#         X_val_xgb = X_val_lgb.copy()
#         X_test_xgb = X_test_lgb.copy()
        
#         xgb_model = xgb.XGBClassifier(**xgb_params)
#         xgb_model.fit(
#             X_train_xgb, y_train,
#             eval_set=[(X_val_xgb, y_val)],
#             verbose=False,
#             early_stopping_rounds=100
#         )
        
#         oof_xgb[val_idx] = xgb_model.predict_proba(X_val_xgb)[:, 1]
#         test_xgb += xgb_model.predict_proba(X_test_xgb)[:, 1] / N_SPLITS
        
#         xgb_score = roc_auc_score(y_val, oof_xgb[val_idx])
#         xgb_scores.append(xgb_score)
#         print(f"  XGBoost   Fold {fold} AUC: {xgb_score:.5f}")
    
#     # ==========================================================================
#     # RESULTS & ENSEMBLE OPTIMIZATION
#     # ==========================================================================
    
#     print("\n" + "="*70)
#     print("INDIVIDUAL MODEL PERFORMANCE")
#     print("="*70)
    
#     lgb_cv = roc_auc_score(y, oof_lgb)
#     xgb_cv = roc_auc_score(y, oof_xgb)
#     cat_cv = roc_auc_score(y, oof_cat)
    
#     print(f"CatBoost  CV AUC: {cat_cv:.5f} ± {np.std(cat_scores):.5f}")
#     print(f"LightGBM  CV AUC: {lgb_cv:.5f} ± {np.std(lgb_scores):.5f}")
#     print(f"XGBoost   CV AUC: {xgb_cv:.5f} ± {np.std(xgb_scores):.5f}")
    
#     # Find optimal weights
#     print("\n" + "="*70)
#     print("ENSEMBLE OPTIMIZATION")
#     print("="*70)
    
#     best_score = 0
#     best_weights = None
    
#     # Test different weight combinations
#     weight_combinations = []
#     for w1 in np.arange(0, 1.1, 0.1):
#         for w2 in np.arange(0, 1.1 - w1, 0.1):
#             w3 = round(1 - w1 - w2, 1)
#             if w3 < 0 or w3 > 1:
#                 continue
            
#             oof_blend = (w1 * oof_cat + w2 * oof_lgb + w3 * oof_xgb)
#             score = roc_auc_score(y, oof_blend)
#             weight_combinations.append((score, (w1, w2, w3)))
            
#             if score > best_score:
#                 best_score = score
#                 best_weights = (w1, w2, w3)
    
#     # Show top 5 weight combinations
#     weight_combinations.sort(reverse=True)
#     print("\nTop 5 weight combinations (CAT, LGB, XGB):")
#     for i, (score, weights) in enumerate(weight_combinations[:5], 1):
#         print(f"  {i}. Weights {weights} -> AUC: {score:.5f}")
    
#     # Final predictions
#     w1, w2, w3 = best_weights
#     oof_final = w1 * oof_cat + w2 * oof_lgb + w3 * oof_xgb
#     test_final = w1 * test_cat + w2 * test_lgb + w3 * test_xgb
    
#     # Simple average for comparison
#     oof_simple = (oof_cat + oof_lgb + oof_xgb) / 3
#     test_simple = (test_cat + test_lgb + test_xgb) / 3
#     simple_score = roc_auc_score(y, oof_simple)
    
#     print(f"\nSimple Average CV AUC: {simple_score:.5f}")
#     print(f"Optimized Ensemble CV AUC: {best_score:.5f}")
#     print(f"Improvement over best single: +{best_score - max(cat_cv, lgb_cv, xgb_cv):.5f}")
    
#     # Save submission
#     submission = pd.DataFrame({
#         'id': test_ids,
#         'loan_paid_back': test_final
#     })
    
#     #submission.to_csv('optimized_ensemble_submission.csv', index=False)
#     submission.to_csv('submission.csv', index=False)
#     print(f"\n✅ Submission saved: 'optimized_ensemble_submission.csv'")
    
#     return {
#         'cv_score': best_score,
#         'weights': best_weights,
#         'predictions': test_final,
#         'individual_scores': {
#             'catboost': cat_cv,
#             'lightgbm': lgb_cv,
#             'xgboost': xgb_cv
#         }
#     }

# # Run the optimized ensemble
# results = run_optimized_ensemble(df_train, df_test)

# print("\n" + "="*70)
# print("EXPECTED OUTCOME:")
# print("="*70)
# print("• CatBoost alone: ~0.9236")
# print("• Ensemble should achieve: ~0.924-0.925")
# print("• To reach 0.926-0.927: Consider meta-learning/stacking")


# y = df_train[target_col]
# test_ids = df_test['id']

# df_train_fe = df_train.drop(['id', target_col], axis=1)
# df_test_fe = df_test.drop(['id'], axis=1)



# X = df_train_fe.copy()
# X_test = df_test_fe.copy()


# y = df_train[target_col]
# test_ids = df_test['id']

# X = df_train_enhanced.drop(['id', target_col], axis=1)
# X_test = df_test_enhanced.drop(['id'], axis=1)


y = df_train[target_col]
test_ids = df_test['id']

X = df_train.drop(['id', target_col], axis=1)
X_test = df_test.drop(['id'], axis=1)


numerical_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ['id', 'loan_paid_back']]
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()


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



# from sklearn.model_selection import train_test_split

# # Split into train/validation (or test)
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y,
#     test_size=0.2,        # 20% for validation
#     random_state=42,      # reproducibility
#     stratify=y            # keeps same class ratio (important for classification)
# )


# from catboost import CatBoostClassifier

# model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=categorical_features)

# #  See all used parameters (includes defaults)
# print(model.get_all_params())



# from catboost import CatBoostClassifier, Pool
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score

## 1. Setup Cross-Validation
N_SPLITS = 5 # You can change this    
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Create arrays to store results
val_scores = []
oof_preds = np.zeros(len(X_test)) # Out-of-fold predictions for the test set
# oof_train_preds = np.zeros(len(X)) # Optional: To store validation preds

## 2. Start the Cross-Validation Loop
# (This continues from your provided loop structure)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*60}")
    
    # Get fold data (using your variables)
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # --- Inside the loop ---
    
    # 1. Initialize the model
    # We pass the params dictionary directly
    model = CatBoostClassifier(**catboost_params)
    
    # 2. Fit the model
    # We provide the categorical features list
    # The validation set is used for early stopping
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              cat_features=categorical_features
              #use_best_model=True
             )
    
    # 3. Make predictions on the validation set
    # We use predict_proba to get probabilities for AUC
    val_preds = model.predict_proba(X_val)[:, 1]
    
    # 4. Score the model
    fold_auc = roc_auc_score(y_val, val_preds)
    val_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

    # *** ADD FEATURE IMPORTANCE HERE (only for first fold) ***
    if fold == 0:  # Only print for first fold
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE (Fold 1)")
        print(f"{'='*60}")
        feature_importance = model.get_feature_importance()
        feature_names = X_train.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        print(importance_df.head(20))
        print(f"{'='*60}\n")
    
    # 5. Make predictions on the test set
    # We average the predictions from each fold
    oof_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # Optional: Store validation predictions
    # oof_train_preds[val_idx] = val_preds


# After the loop ends, before final results
print(f"\n{'='*60}")
print(f"FEATURE IMPORTANCE (Final Fold)")
print(f"{'='*60}")
feature_importance = model.get_feature_importance()
feature_names = X.columns  # Use X, not X_train here since we're outside the loop
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)
print(importance_df.head(20))
print(f"{'='*60}\n")


# Report Final Score
print(f"\n{'='*60}")
print(f"Overall CV AUC: {np.mean(val_scores):.5f} +/- {np.std(val_scores):.5f}")
print(f"{'='*60}")

print(f"\n{'='*60}")
print(f"Base CV AUC: 0.92359 +/- 0.00070")
print(f"{'='*60}")


overall_auc = np.mean(val_scores)
overall_std = np.std(val_scores)
base_auc = 0.92359
base_std = 0.00070

# Report Final Score
print(f"\n{'='*60}")
print(f"Overall CV AUC: {overall_auc:.5f} +/- {overall_std:.5f}")
print(f"Base CV AUC:    {base_auc:.5f} +/- {base_std:.5f}")
print(f"{'='*60}")

# --- Comparison ---
difference = overall_auc - base_auc

# Check if it's an improvement and format the output
if difference > 0:
    print(f"Improvement of: +{difference:.5f}")
elif difference < 0:
    print(f"Regression of: {difference:.5f}")
else:
    print("No change in score.")
    
print(f"{'='*60}")


# Create Submission File
submission_df = pd.DataFrame({'id': test_ids, target_col: oof_preds})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print(submission_df.head())




