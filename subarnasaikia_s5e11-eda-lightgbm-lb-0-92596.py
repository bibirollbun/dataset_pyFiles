import warnings, torch
warnings.filterwarnings('ignore')


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier as XGBC
from lightgbm import LGBMClassifier as LGBMC, log_evaluation, early_stopping
from catboost import CatBoostClassifier as CBC
from sklearn.metrics import *


# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


test_req = False

if test_req :
    print("THIS IS A SYNTAX CHECK RUN")
    nest = 150
    es   = 50
else:
    nest = 12000
    es   = 300


TRAIN_PATH = "/kaggle/input/playground-series-s5e11/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e11/test.csv"
ORIG_PATH = "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/playground-series-s5e11/sample_submission.csv"


# Load datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
orig = pd.read_csv(ORIG_PATH)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Orig shape: {orig.shape}")
print(f"\nTrain columns: {train.columns.tolist()}\n")
print(f"Test columns: {test.columns.tolist()}\n")
print(f"Orig columns: {orig.columns.tolist()}\n")


# Basic info
print("=" * 50)
print("TRAINING DATA INFO")
print("=" * 50)
display(train.info())
print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
display(train.describe())


# Check for missing values
print("Missing values in train:")
display(train.isnull().sum())
print("\nMissing values in test:")
display(test.isnull().sum())
print("\nMissing values in orig:")
display(orig.isnull().sum())


# Target distribution
print("Target distribution:")
print(train['loan_paid_back'].value_counts(normalize=True))

plt.figure(figsize=(8, 5))
train['loan_paid_back'].value_counts().plot(kind='bar', color=['salmon', 'skyblue'])
plt.title('Loan Repayment Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Loan Paid Back (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Numerical features distribution
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                  'loan_amount', 'interest_rate']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].hist(train[col], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{col}', fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

axes[-1].axis('off')
plt.tight_layout()
plt.show()


# Correlation with target
correlations = train[numerical_cols + ['loan_paid_back']].corr()['loan_paid_back'].sort_values(ascending=False)
print("Correlation with target:")
print(correlations)

plt.figure(figsize=(8, 6))
correlations[:-1].plot(kind='barh', color='coral')
plt.title('Feature Correlation with Loan Repayment', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()


# Categorical features analysis
CATS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

print("Categorical feature cardinality:")
for col in CATS:
    print(f"{col}: {train[col].nunique()} unique values")


# Target rate by categorical features (sample for a few)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

sample_cats = ['gender', 'marital_status', 'education_level', 'employment_status']

for i, col in enumerate(sample_cats):
    target_rate = train.groupby(col)['loan_paid_back'].mean().sort_values()
    target_rate.plot(kind='barh', ax=axes[i], color='teal')
    axes[i].set_title(f'Repayment Rate by {col}', fontweight='bold')
    axes[i].set_xlabel('Repayment Rate')

plt.tight_layout()
plt.show()


# Identify feature types
TARGET = 'loan_paid_back'
id_col = 'id'

# Features to drop
drop_cols = [id_col, TARGET]

# Get feature columns
BASE = [col for col in train.columns if col not in drop_cols]

# Separate numerical and categorical
categorical_features = train[BASE].select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = train[BASE].select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Total features: {len(BASE)}")
print(f"Numerical: {len(numerical_features)}")
print(f"Categorical: {len(categorical_features)}")


# https://www.kaggle.com/code/masayakawamata/s5e11-te-xgb-interaction-features?scriptVersionId=272584844&cellId=5

from itertools import combinations

INTER = []

for col1, col2 in combinations(BASE, 2):
    new_col_name = f'{col1}_{col2}'
    INTER.append(new_col_name)
    for df in [train, test, orig]:
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
        
        
print(f'{len(INTER)} Features.')


ORIG = []

for col in BASE:
    # MEAN
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    orig = orig.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    orig = orig.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(len(ORIG), 'Orig Features Created!!')


def create_features(df):
    """Create additional features from existing ones"""
    df = df.copy()
    
    # Income-based features
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['loan_to_income_pct'] = (df['loan_amount'] / df['annual_income']) * 100
    
    # Risk features
    df['high_dti'] = (df['debt_to_income_ratio'] > 0.4).astype(int)
    df['low_credit'] = (df['credit_score'] < 650).astype(int)
    df['high_interest'] = (df['interest_rate'] > 15).astype(int)
    
    # Combined risk score
    df['risk_score'] = df['high_dti'] + df['low_credit'] + df['high_interest']
    
    # Credit score bins
    df['credit_score_bin'] = pd.cut(df['credit_score'], 
                                      bins=[0, 580, 670, 740, 800, 900],
                                      labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
    
    # Income bins
    df['income_bin'] = pd.qcut(df['annual_income'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # Extract grade from grade_subgrade (e.g., 'A1' -> 'A')
    df['grade'] = df['grade_subgrade'].str[0]
    
    return df


NEW = ['income_to_loan_ratio', 'loan_to_income_pct', 'high_dti', 'low_credit', 'high_interest', 'risk_score',
       'credit_score_bin', 'income_bin', 'grade']
CATS += ['credit_score_bin', 'income_bin', 'grade']
# Apply feature engineering
train = create_features(train)
test = create_features(test)
orig = create_features(orig)

print("New features created!")
print(f"Train shape after feature engineering: {train.shape}")


FEATURES = BASE + ORIG + INTER + NEW
print(len(FEATURES), 'Features.')
categorical_features = train[FEATURES].select_dtypes(include=['object', 'category']).columns.tolist()
print(len(categorical_features), 'categorical_features.')


# Keep only necessary columns from orig (same structure as train)
orig_trimmed = orig[[*FEATURES, TARGET]].copy()

# Combine both dataframes
combined = pd.concat([train, orig_trimmed], ignore_index=True)

# Drop duplicate rows based on FEATURES + TARGET
combined = combined.drop_duplicates(subset=FEATURES + [TARGET], keep='first').reset_index(drop=True)


# Encode categorical variables
label_encoders = {}

for col in CATS:
    le = LabelEncoder()
    # Fit on combined train+test to handle unseen categories
    combined_new = pd.concat([combined[col], test[col]], axis=0)
    le.fit(combined_new.astype(str))
    
    combined[col] = le.transform(combined[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    
    label_encoders[col] = le

print("Categorical encoding completed!")


# Prepare X and y

# 4️⃣ Extract final X and y
X = combined[FEATURES]
y = combined[TARGET]
X_test = test[FEATURES]


print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")


from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder that supports multiple aggregation functions,
    internal cross-validation for leakage prevention, and smoothing.

    Parameters
    ----------
    cols_to_encode : list of str
        List of column names to be target encoded.

    aggs : list of str, default=['mean']
        List of aggregation functions to apply. Any function accepted by
        pandas' `.agg()` method is supported, such as:
        'mean', 'std', 'var', 'min', 'max', 'skew', 'nunique', 
        'count', 'sum', 'median'.
        Smoothing is applied only to the 'mean' aggregation.

    cv : int, default=5
        Number of folds for cross-validation in fit_transform.

    smooth : float or 'auto', default='auto'
        The smoothing parameter `m`. A larger value puts more weight on the 
        global mean. If 'auto', an empirical Bayes estimate is used.
        
    drop_original : bool, default=False
        If True, the original columns to be encoded are dropped.
    """
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        """
        Learn mappings from the entire dataset.
        These mappings are used for the transform method on validation/test data.
        """
        temp_df = X.copy()
        temp_df['target'] = y

        # Learn global statistics for each aggregation
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        # Learn category-specific mappings
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        
        return self

    def transform(self, X):
        """
        Apply learned mappings to the data.
        Unseen categories are filled with global statistics.
        """
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
        
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit and transform the data using internal cross-validation to prevent leakage.
        """
        # First, fit on the entire dataset to get global mappings for transform method
        self.fit(X, y)

        # Initialize an empty DataFrame to store encoded features
        encoded_features = pd.DataFrame(index=X.index)
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                # --- Calculate mappings only on the training part of the fold ---
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    
                    # Calculate global stat for this fold
                    fold_global_stat = y_train.agg(agg_func)
                    
                    # Calculate category stats for this fold
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    # --- Apply smoothing only for 'mean' aggregation ---
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        
                        m = self.smooth
                        if self.smooth == 'auto':
                            # Empirical Bayes smoothing
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0  # No smoothing if no variance between groups
                        
                        # Apply smoothing formula
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    
                    # Store encoded values for the validation fold
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        # Merge with original DataFrame
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
            
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed


# # LightGBM parameters
# params = {
#     'objective': 'binary',
#     'metric': 'auc',
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.05,
#     'num_leaves': 31,
#     'max_depth': -1,
#     'min_child_samples': 20,
#     'subsample': 0.8,
#     'subsample_freq': 1,
#     'colsample_bytree': 0.8,
#     'reg_alpha': 0.1,
#     'reg_lambda': 0.1,
#     'random_state': SEED,
#     'n_jobs': -1,
#     'verbose': -1
# }

# Cross-validation setup
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)




# --- Configuration / containers ---
Mdl_Master = {
    'XGB1C': [
        XGBC(**{
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "learning_rate": 0.01,
            "n_estimators": nest,
            "max_depth": 8,
            "subsample": 0.90,
            "colsample_bytree": 0.75,
            "reg_lambda": 0.75,
            "reg_alpha": 0.001,
            "verbosity": 0,
            "random_state": 42,
            "enable_categorical": True,
            "early_stopping_rounds": es,
        }),
        {"verbose": 0}  # extra fit params for this model (if used)
    ],

    'LGBM1C': [
        LGBMC(**{
            "objective": "binary",
            "eval_metric": "auc",
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "learning_rate": 0.01,
            "n_estimators": nest,
            "max_depth": 7,
            "subsample": 0.90,
            "colsample_bytree": 0.60,
            "reg_lambda": 1.25,
            "reg_alpha": 0.001,
            "verbosity": -1,  # suppress verbose LGBM logs
            "random_state": 42,
        }),
        {
            "callbacks": [
                log_evaluation(period=100),  # print evaluation every 100 rounds
                early_stopping(es, verbose=False)
            ]
        }
    ],

    'LGBM2C': [
        LGBMC(**{
            "objective": "binary",
            "data_sample_strategy": "goss",
            "eval_metric": "auc",
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "learning_rate": 0.01,
            "n_estimators": nest,
            "max_depth": 6,
            "subsample": 0.825,
            "colsample_bytree": 0.55,
            "reg_lambda": 0.85,
            "reg_alpha": 0.001,
            "verbosity": -1,
            "random_state": 42,
        }),
        {
            "callbacks": [
                log_evaluation(period=100),
                early_stopping(es, verbose=False)
            ]
        }
    ],
}


# Storage
n_samples = len(X)
oof_predictions = np.zeros(n_samples, dtype=float)  # final ensemble oof
test_predictions = None  # will init after we know X_test length
feature_importance = pd.DataFrame()
cv_scores_model = {k: [] for k in Mdl_Master.keys()}  # per-model CV scores
ensemble_cv_scores = []


print("Starting cross-validation training...")
print("=" * 60)

# silence non-critical warnings (be careful with this globally; you might want a context)
warnings.filterwarnings("ignore")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}/{N_FOLDS}")

    # Split data
    X_train_fold, X_val_fold = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train_fold, y_val_fold = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
    X_test_fold = test[FEATURES].copy()  # keep same name

    # Target encoding (fit only on train part)
    TE = TargetEncoder(cols_to_encode=INTER, cv=5, smooth='auto', aggs=['mean'], drop_original=True)
    X_train_fold = TE.fit_transform(X_train_fold, y_train_fold)
    X_val_fold = TE.transform(X_val_fold)
    X_test_transformed = TE.transform(X_test_fold)

    # Initialize containers for this fold
    per_model_val_preds = []
    per_model_test_preds = []

    # initialize test_predictions vector once (we know length of X_test_transformed)
    if test_predictions is None:
        test_predictions = np.zeros(len(X_test_transformed), dtype=float)

    for method, (model, fit_params) in Mdl_Master.items():
        # Use a local copy to avoid accidental mutation
        local_fit_params = dict(fit_params)

        # If callbacks/logging present in fit_params, we've already set log_evaluation(period=100)
        # Fit model (safely pass eval_set and any callbacks)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            **local_fit_params
        )

        # Probabilistic predictions for validation and test
        val_pred = pd.Series(model.predict_proba(X_val_fold)[:, 1], index=val_idx, name=method)
        test_pred = pd.Series(model.predict_proba(X_test_transformed)[:, 1], name=method)

        per_model_val_preds.append(val_pred)
        per_model_test_preds.append(test_pred)

        # model-level AUC
        model_auc = roc_auc_score(y_val_fold, val_pred.values)
        cv_scores_model[method].append(model_auc)
        print(f"Fold {fold}_{method} AUC: {model_auc:.6f}")
        print(f"---> Model {method} fitted successfully")

        # Feature importance: be robust to different shapes/names after TE
        feat_names = list(X_train_fold.columns)
        fi = getattr(model, "feature_importances_", None)
        if fi is not None and len(fi) == len(feat_names):
            fold_importance = pd.DataFrame({
                'feature': feat_names,
                'importance': fi,
                'fold': fold,
                'model': method
            })
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        else:
            # if model doesn't provide importances or shapes mismatch, skip/notify
            print(f"Warning: feature_importances_ not available or shape mismatch for {method}; skipping FI for this model.")

    # combine per-model preds into DataFrame indexed by val_idx
    val_preds_df = pd.concat(per_model_val_preds, axis=1)
    test_preds_df = pd.concat(per_model_test_preds, axis=1)

    # Ensemble strategy: simple mean across models (you can change to weighted average)
    ensemble_val = val_preds_df.mean(axis=1)
    ensemble_test = test_preds_df.mean(axis=1)

    # store ensemble OOF predictions
    oof_predictions[val_idx] = ensemble_val.values

    # accumulate test predictions (averaged across folds)
    test_predictions += ensemble_test.values / N_FOLDS

    # also record ensemble fold AUC
    ens_auc = roc_auc_score(y_val_fold, ensemble_val.values)
    ensemble_cv_scores.append(ens_auc)
    print(f"Fold {fold} Ensemble AUC: {ens_auc:.6f}")

print("\n" + "=" * 60)



# Summaries
# per-model mean+std
for method, scores in cv_scores_model.items():
    if scores:
        print(f"{method} CV AUC: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")
# ensemble summary
print(f"Ensemble Mean CV AUC: {np.mean(ensemble_cv_scores):.6f} (+/- {np.std(ensemble_cv_scores):.6f})")

print("=" * 60)

# Optionally: create a DataFrame for oof predictions and check length
oof_df = pd.DataFrame({'oof_pred': oof_predictions, 'target': y.values})
assert len(oof_df) == n_samples, "OOF size mismatch"


# Overall OOF AUC
oof_auc = roc_auc_score(y, oof_predictions)
print(f"Out-of-Fold AUC: {oof_auc:.6f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y, oof_predictions)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {oof_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Loan Default Prediction', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Feature importance
importance_df = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(10, 8))
top_n = 20
plt.barh(range(top_n), importance_df['importance'][:top_n], color='steelblue')
plt.yticks(range(top_n), importance_df['feature'][:top_n])
plt.xlabel('Average Importance (Gain)', fontsize=12)
plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))


# Create submission dataframe
submission = pd.DataFrame({
    'id': test[id_col],
    'loan_paid_back': test_predictions
})

# Verify submission format
print("Submission shape:", submission.shape)
print("\nFirst few predictions:")
print(submission.head(10))
print("\nPrediction statistics:")
print(submission['loan_paid_back'].describe())

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")

