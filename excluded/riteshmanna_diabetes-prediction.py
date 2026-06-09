import warnings
warnings.simplefilter('ignore')

import pandas as pd, numpy as np, gc

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Orig Shape:', orig.shape)


train.info()


test.info()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

TARGET='diagnosed_diabetes'

# --- Train Dataset ---
sns.countplot(data=train, x=TARGET, ax=ax[0], palette='viridis')
ax[0].set_title(f'Train: {TARGET} Distribution')

# Calculate total for percentage
total_train = len(train)

for p in ax[0].patches:
    height = p.get_height()
    percentage = (height / total_train) * 100
    # Annotate with: Count (Percentage%)
    ax[0].annotate(f'{height:,} ({percentage:.1f}%)', 
                   (p.get_x() + p.get_width() / 2., height), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# --- Original Dataset
sns.countplot(data=orig, x=TARGET, ax=ax[1], palette='viridis')
ax[1].set_title(f'Original: {TARGET} Distribution')

# Calculate total for percentage
total_orig = len(orig)

for p in ax[1].patches:
    height = p.get_height()
    percentage = (height / total_orig) * 100
    # Annotate with: Count (Percentage%)
    ax[1].annotate(f'{height:,} ({percentage:.1f}%)', 
                   (p.get_x() + p.get_width() / 2., height), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()


BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = train.select_dtypes('object').columns.to_list()
NUMS = [col for col in BASE if col not in CATS]


corr_matrix = train[NUMS + [TARGET]].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
            cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix (Features vs Target)')
plt.show()


#Target encoding through original data

ORIG = []

for col in BASE:
    # MEAN
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

print(f'{len(ORIG)} ORIG Features Created.')


for col in ORIG:
    if 'mean' in col:
        train[col] = train[col].fillna(orig[TARGET].mean())
        test[col] = test[col].fillna(orig[TARGET].mean())
    else:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)


FEATURES = BASE + ORIG
print(len(FEATURES), 'Features.')


X = train[FEATURES]
y = train[TARGET]


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

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


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc

# -----------------------------------------------------
# 0. Helper: Memory Reducer
# -----------------------------------------------------
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

X = reduce_mem_usage(X)
test = reduce_mem_usage(test)
gc.collect()

# -----------------------------------------------------
# 4.3 Training Loop (XGBoost)
# -----------------------------------------------------

# Initialize arrays
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

# Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Select TE Columns (Exclude binary features)
TE_COLS = [col for col in NUMS if train[col].nunique() > 2]
print(f"Target Encoding applied to {len(TE_COLS)} features.")

# XGBoost Parameters
xgb_params = {
    'n_estimators': 20000,
    'learning_rate': 0.01,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'device': 'cuda',           # GPU (optional)
    'enable_categorical': True  # Native Categorical Support
}

print(f"Starting Training...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    # 1. Split Data
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
    X_test_fold = test[FEATURES].copy() 
    
    # -----------------------------------------------------
    # A. Target Encoding (Augment Numerical Features)
    # -----------------------------------------------------
    if len(TE_COLS) > 0:
        TE = TargetEncoder(cols_to_encode=TE_COLS, cv=5, smooth='auto', aggs=['mean', 'count'], drop_original=False)
        X_train = TE.fit_transform(X_train, y_train)
        X_val = TE.transform(X_val)
        X_test_fold = TE.transform(X_test_fold)
    
    # -----------------------------------------------------
    # B. Factorize CATS & Prepare for Native Support
    # -----------------------------------------------------
    for c in CATS:
        # 1. Factorize (Returns NumPy Array)
        combined = pd.concat([X_train[c], X_val[c], X_test_fold[c]])
        combined_encoded, _ = combined.factorize()
        
        # 2. Assign back to DataFrame to convert to Series
        X_train[c] = combined_encoded[:len(X_train)]
        X_val[c] = combined_encoded[len(X_train):len(X_train)+len(X_val)]
        X_test_fold[c] = combined_encoded[len(X_train)+len(X_val):]

        # 3. Cast to Category
        X_train[c] = X_train[c].astype('category')
        X_val[c] = X_val[c].astype('category')
        X_test_fold[c] = X_test_fold[c].astype('category')

    # -----------------------------------------------------
    # C. Train Model (XGBClassifier)
    # -----------------------------------------------------
    model = XGBClassifier(**xgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=200,
        verbose=500
    )
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test_fold)[:, 1] / kf.get_n_splits()
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f"Fold {fold+1} AUC: {fold_score:.5f}")

    if len(TE_COLS) > 0: del TE
    gc.collect()

print("-" * 30)
print(f"OOF AUC: {roc_auc_score(y, oof_preds):.5f}")


# 1. Create Submission DataFrame
sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
sub[TARGET] = test_preds
sub.to_csv('submission.csv', index=False)

# 2. Save OOF Predictions (for Ensembling)
# OOF dataframe creating ensures we match the correct IDs
oof_df = pd.DataFrame()
oof_df['id'] = train['id']
oof_df[TARGET] = y
oof_df['pred'] = oof_preds
oof_df.to_csv('oof_predictions.csv', index=False)

print('Submission and OOF files saved successfully.')
print(f'Submission Shape: {sub.shape}')

# 3. Sanity Check: Distribution Plot
plt.figure(figsize=(10, 5))
sns.kdeplot(oof_df['pred'], label='OOF Predictions (Train)', fill=True, color='blue', alpha=0.3)
sns.kdeplot(sub[TARGET], label='Test Predictions', fill=True, color='orange', alpha=0.3)
plt.title('Distribution of Predictions: OOF vs Test')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
plt.show()

sub.head()




