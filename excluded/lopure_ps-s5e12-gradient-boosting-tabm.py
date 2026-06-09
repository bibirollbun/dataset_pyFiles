import pandas as pd, numpy as np, gc

train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Orig Shape:', orig.shape)

train.head(3)


TARGET = 'diagnosed_diabetes'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = train.select_dtypes('object').columns.to_list()
NUMS = [col for col in BASE if col not in CATS]
print(f'{len(BASE)} Base Features:{BASE}')


print('NaN Count:', train[CATS].isnull().sum().sum(), '\n')
print(train[CATS].nunique(),'\n')

for col in CATS:
    # Get the unique values for the current column
    unique_values = train[col].unique()
    
    # Print the column name and its unique values
    print(f"\nColumn Name: {col}")
    print(unique_values)
train[CATS].head(3)


print('NaN Count:', train[NUMS].isnull().sum().sum(), '\n')
print(train[NUMS].nunique(),'\n')
train[NUMS].head(3)



ORIG = []

for col in BASE:
    # MEAN
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

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
import warnings
warnings.filterwarnings("ignore")

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


from itertools import combinations
import pandas as pd
def add_interaction_features(df):
    # 1. Lifestyle interaction (Activity-Screen Balance)
    df['lifestyle_balance'] = df['physical_activity_minutes_per_week'] / (df['screen_time_hours_per_day'] + 1)
    
    # 2. Health metrics (Body composition risk)
    df['body_composition_risk'] = df['bmi'] * df['waist_to_hip_ratio']
    
    # 3. Blood pressure (Pulse pressure ratio)
    df['bp_ratio'] = df['systolic_bp'] / df['diastolic_bp']
    
    # 4. Cholesterol ratios
    df['insulin_resistance_marker'] = df['triglycerides'] / df['hdl_cholesterol']
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / df['hdl_cholesterol']
    
    # 5. Lipid interaction
    df['lipid_interaction'] = (df['triglycerides'] * df['ldl_cholesterol']) / df['hdl_cholesterol']
    
    # 6. Age interactions
    df['age_bmi'] = df['age'] * df['bmi']
    df['age_systolic_bp'] = df['age'] * df['systolic_bp']
    
    return df

def create_interaction_features(train_df, val_df, test_df, cat_cols):
    """
    Creates interactions for Train, Val, and Test sets based on Train statistics.
    Includes robustness for unseen categories and column filtering.
    """
    train_processed = train_df.copy()
    val_processed = val_df.copy()
    test_processed = test_df.copy()

    # --- Step 1: Safe Label Mapping (Robust to unseen labels) ---
    active_cats = [c for c in cat_cols if c in train_processed.columns]
    
    for col in active_cats:
        # Create mapping from train only
        unique_vals = train_processed[col].unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        
        # Apply mapping, fill unseen with a new index (len of mapping)
        train_processed[col] = train_processed[col].map(mapping)
        val_processed[col] = val_processed[col].map(mapping).fillna(len(mapping)).astype(int)
        test_processed[col] = test_processed[col].map(mapping).fillna(len(mapping)).astype(int)

    # --- Step 2: Calculate Cardinality from Train ---
    sizes = {col: train_processed[col].nunique() for col in active_cats}

    # --- Step 3 & 4: Create and Add Interactions ---
    pairs = list(combinations(active_cats, 2))
    new_cols_train, new_cols_val, new_cols_test = {}, {}, {}
    
    for c1, c2 in pairs:
        name = f"inter_{c1}_{c2}"
        # Formula: Feature_A * Size_B + Feature_B
        # This creates a unique integer for every unique combination of A and B
        new_cols_train[name] = train_processed[c1] * sizes[c2] + train_processed[c2]
        new_cols_val[name] = val_processed[c1] * sizes[c2] + val_processed[c2]
        new_cols_test[name] = test_processed[c1] * sizes[c2] + test_processed[c2]

    # Batch concatenate for memory efficiency
    if new_cols_train:
        train_processed = pd.concat([train_processed, pd.DataFrame(new_cols_train, index=train_df.index)], axis=1)
        val_processed = pd.concat([val_processed, pd.DataFrame(new_cols_val, index=val_df.index)], axis=1)
        test_processed = pd.concat([test_processed, pd.DataFrame(new_cols_test, index=test_df.index)], axis=1)
    
    return train_processed, val_processed, test_processed
    
# Apply the features
X = add_interaction_features(X)
test = add_interaction_features(test)


NEW_FEATURES = [
    'lifestyle_balance', 'body_composition_risk', 'bp_ratio', 
    'insulin_resistance_marker', 'ldl_hdl_ratio', 'lipid_interaction',
    'age_bmi', 'age_systolic_bp'
]
FEATURES = BASE + ORIG + NEW_FEATURES
FEATURES




print(X.shape)
CATS = [c for c in CATS if c in FEATURES]  
NUMS = [col for col in BASE if col not in CATS]



def create_frequency_features(df, df_val, df_test, active_nums, active_cats):
    """
    Creates frequency and quantile binning features for Train, Val, and Test.
    Uses 'df' (Train) as the reference for mapping and bin edges.
    """
    # Temporary storage to avoid fragmentation
    train_new, val_new, test_new = pd.DataFrame(index=df.index), pd.DataFrame(index=df_val.index), pd.DataFrame(index=df_test.index)

    for col in active_nums + active_cats:
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        train_new[f"{col}_freq"] = df[col].map(freq)
        val_new[f"{col}_freq"] = df_val[col].map(freq).fillna(freq.mean())
        test_new[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # --- Quantile binning ---
        if col in active_nums:
            for q in [5, 10, 15]:
                try:
                    # Get bins from Train
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    train_new[f"{col}_bin{q}"] = train_bins
                    # Apply Train bins to Val and Test
                    val_new[f"{col}_bin{q}"] = pd.cut(df_val[col], bins=bins, labels=False, include_lowest=True).fillna(-1)
                    test_new[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True).fillna(-1)
                except Exception:
                    train_new[f"{col}_bin{q}"], val_new[f"{col}_bin{q}"], test_new[f"{col}_bin{q}"] = 0, 0, 0

    # Combine
    df = pd.concat([df, train_new], axis=1)
    df_val = pd.concat([df_val, val_new], axis=1)
    df_test = pd.concat([df_test, test_new], axis=1)

    return df, df_val, df_test


def remove_correlated_features(X_train, X_val, X_test, threshold=0.95):
    """
    Identifies and removes highly correlated features based on X_train.
    """
    # Calculate correlation matrix only on numerical columns
    corr_matrix = X_train.select_dtypes(include=[np.number]).corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"   ğŸ“‰ Removing {len(to_drop)} highly correlated features...")
        X_train = X_train.drop(columns=to_drop)
        X_val = X_val.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
        
    return X_train, X_val, X_test, to_drop
    


def create_division_features(df_train, df_val, df_test):
    """
    Automatically detects aggregated columns and creates 'per' features.
    Handles division by zero and ensures consistency across folds.
    """
    train_new, val_new, test_new = pd.DataFrame(index=df_train.index), pd.DataFrame(index=df_val.index), pd.DataFrame(index=df_test.index)
    
    # Identify column groups (e.g., TE1_wc)
    # This assumes your columns follow a pattern like 'prefix_stat'
    prefixes = set(['_'.join(c.split('_')[:-1]) for c in df_train.columns if any(s in c for s in ['_count', '_nunique', '_std', '_mean'])])

    for p in prefixes:
        cols = {
            'count': f"{p}_count",
            'nunique': f"{p}_nunique",
            'std': f"{p}_std",
            'mean': f"{p}_mean"
        }
        
        # Check if necessary columns exist for the ratio
        # 1. COUNT PER NUNIQUE
        if cols['count'] in df_train.columns and cols['nunique'] in df_train.columns:
            name = f"{p}_count_per_nunique"
            for df_src, df_dest in zip([df_train, df_val, df_test], [train_new, val_new, test_new]):
                # Use .div to handle division by zero (results in inf) then replace with 0
                df_dest[name] = df_src[cols['count']].div(df_src[cols['nunique']]).replace([np.inf, -np.inf], 0).fillna(0)

        # 2. STD PER COUNT (Coefficient of Variation variant)
        if cols['std'] in df_train.columns and cols['count'] in df_train.columns:
            name = f"{p}_std_per_count"
            for df_src, df_dest in zip([df_train, df_val, df_test], [train_new, val_new, test_new]):
                df_dest[name] = df_src[cols['std']].div(df_src[cols['count']]).replace([np.inf, -np.inf], 0).fillna(0)

        # 3. MEAN PER NUNIQUE (Density feature)
        if cols['mean'] in df_train.columns and cols['nunique'] in df_train.columns:
            name = f"{p}_mean_per_nunique"
            for df_src, df_dest in zip([df_train, df_val, df_test], [train_new, val_new, test_new]):
                df_dest[name] = df_src[cols['mean']].div(df_src[cols['nunique']]).replace([np.inf, -np.inf], 0).fillna(0)

    # Concatenate and return
    return pd.concat([df_train, train_new], axis=1), \
           pd.concat([df_val, val_new], axis=1), \
           pd.concat([df_test, test_new], axis=1)


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


from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


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
# 4.3 Training Loop (LightGBM)
# -----------------------------------------------------

# Initialize arrays
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

# Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Select TE Columns (Exclude binary features)
TE_COLS = [col for col in NUMS if train[col].nunique() > 2]
print(f"Target Encoding applied to {len(TE_COLS)} features.")

# LightGBM Parameters
lgb_params = {
    'n_estimators': 20000,
    'learning_rate': 0.01,
    'max_depth': 4,            # LGBM handles depth better with num_leaves
    'num_leaves': 31,           # Standard starting point
    'subsample': 0.5,
    'colsample_bytree': 0.2,
    'lambda_l2': 15.0,
    'lambda_l1': 10.0,
    'random_state': 42,
    'n_jobs': -1,
    'metric': 'auc',            # Changed from eval_metric
    'device': 'gpu',            # Changed from 'cuda' to 'gpu' for LGBM
    'verbosity': -1             # Reduce logging noise
}

print(f"Starting LightGBM Training...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    # 1. Split Data
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
    X_test_fold = test[FEATURES].copy() 

    # Filter helper lists
    active_nums = [c for c in NUMS if c in X_train.columns]
    active_cats = [c for c in CATS if c in X_train.columns]
    
    # --- A. Feature Engineering (Frequency & Interactions) ---
    X_train, X_val, X_test_fold = create_frequency_features(X_train, X_val, X_test_fold, active_nums, active_cats)
    
    # --- B. Target Encoding ---
    if len(TE_COLS) > 0:
        active_te = [c for c in TE_COLS if c in X_train.columns]
        TE = TargetEncoder(cols_to_encode=active_te, cv=5, smooth='auto', aggs=['mean', 'count'], drop_original=False)
        X_train = TE.fit_transform(X_train, y_train)
        X_val = TE.transform(X_val)
        X_test_fold = TE.transform(X_test_fold)
    # 2. Generate Division Features
    X_train, X_val, X_test_fold = create_division_features(X_train, X_val, X_test_fold)

    # Ensure BASE_COLS weren't dropped (optional safety check)
    # dropped = [d for d in dropped if d not in BASE_COLS]
    
    # --- D. Factorize & Native Categorical Support ---
    # Re-calculate CATS in case any were dropped due to correlation
    remaining_cats = [c for c in active_cats if c in X_train.columns]
    for c in remaining_cats:
        combined = pd.concat([X_train[c], X_val[c], X_test_fold[c]])
        combined_encoded, _ = combined.factorize()
        X_train[c] = combined_encoded[:len(X_train)]
        X_val[c] = combined_encoded[len(X_train):len(X_train)+len(X_val)]
        X_test_fold[c] = combined_encoded[len(X_train)+len(X_val):]
        X_train[c] = X_train[c].astype('category')
        X_val[c] = X_val[c].astype('category')
        X_test_fold[c] = X_test_fold[c].astype('category')

    X_train = reduce_mem_usage(X_train)
    X_val = reduce_mem_usage(X_val)
    
    # -----------------------------------------------------
    # C. Train Model (LGBMClassifier)
    # -----------------------------------------------------
    model = LGBMClassifier(**lgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=200),
            log_evaluation(period=500) # Replaces the 'verbose' argument in fit
        ]
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
# 1. Create Submission DataFrame
sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
sub[TARGET] = test_preds
sub.to_csv('lgbml1&l2subm.csv', index=False)

# 2. Save OOF Predictions (for Ensembling)
# OOF dataframe creating ensures we match the correct IqDs
oof_df = pd.DataFrame()
oof_df['id'] = train['id']
oof_df[TARGET] = y
oof_df['pred'] = oof_preds
oof_df.to_csv('oof_predictionslgbml1&l2.csv', index=False)

print('Submission and OOF files saved successfully.')
print(f'Submission Shape: {sub.shape}')
print(f"OOF AUC: {roc_auc_score(y, oof_preds):.5f}")


###### from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
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
    'subsample': 0.5,
    'colsample_bytree': 0.2,
    'lambda': 15,
    'alpha': 10,
    'gamma' : 3,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 200,
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

    # Filter helper lists
    active_nums = [c for c in NUMS if c in X_train.columns]
    active_cats = [c for c in CATS if c in X_train.columns]
    
    # --- A. Feature Engineering (Frequency & Interactions) ---
    X_train, X_val, X_test_fold = create_frequency_features(X_train, X_val, X_test_fold, active_nums, active_cats)
    X_train, X_val, X_test_fold = create_interaction_features(X_train, X_val, X_test_fold, active_cats)
    
    # --- B. Target Encoding ---
    if len(TE_COLS) > 0:
        active_te = [c for c in TE_COLS if c in X_train.columns]
        TE = TargetEncoder(cols_to_encode=active_te, cv=5, smooth='auto', aggs=['mean', 'count'], drop_original=False)
        X_train = TE.fit_transform(X_train, y_train)
        X_val = TE.transform(X_val)
        X_test_fold = TE.transform(X_test_fold)
    # 2. Generate Division Features
    X_train, X_val, X_test_fold = create_division_features(X_train, X_val, X_test_fold)

    # Ensure BASE_COLS weren't dropped (optional safety check)
    # dropped = [d for d in dropped if d not in BASE_COLS]
    
    # --- D. Factorize & Native Categorical Support ---
    # Re-calculate CATS in case any were dropped due to correlation
    remaining_cats = [c for c in active_cats if c in X_train.columns]
    for c in remaining_cats:
        combined = pd.concat([X_train[c], X_val[c], X_test_fold[c]])
        combined_encoded, _ = combined.factorize()
        X_train[c] = combined_encoded[:len(X_train)]
        X_val[c] = combined_encoded[len(X_train):len(X_train)+len(X_val)]
        X_test_fold[c] = combined_encoded[len(X_train)+len(X_val):]
        X_train[c] = X_train[c].astype('category')
        X_val[c] = X_val[c].astype('category')
        X_test_fold[c] = X_test_fold[c].astype('category')

    X_train = reduce_mem_usage(X_train)
    X_val = reduce_mem_usage(X_val)
    
    # --- E. Train XGBoost ---
    model = XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=500 # Set to False or 500
    )
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test_fold)[:, 1] / kf.get_n_splits()
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")
    gc.collect()

print("-" * 30)
# 1. Create Submission DataFrame
sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
sub[TARGET] = test_preds
sub.to_csv('xgbl1&l2subm.csv', index=False)

# 2. Save OOF Predictions (for Ensembling)
# OOF dataframe creating ensures we match the correct IqDs
oof_df = pd.DataFrame()
oof_df['id'] = train['id']
oof_df[TARGET] = y
oof_df['pred'] = oof_preds
oof_df.to_csv('oof_predictionsxgbl1&l2.csv', index=False)

print('Submission and OOF files saved successfully.')
print(f'Submission Shape: {sub.shape}')
print(f"ğŸ�� FINAL OOF AUC: {roc_auc_score(y, oof_preds):.5f}")


# -----------------------------------------------------
# 2. Analyze Results
# -----------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
imp_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 12))
sns.barplot(data=imp_df.head(50), x='Importance', y='Feature', palette='viridis')
plt.title('Top 40 Feature Importances (LGBMBoost - Last Fold)')
plt.xlabel('Importance (Gain)')
plt.ylabel('Feature')
plt.show()


!pip install -qq pytabkit


import warnings
warnings.simplefilter('ignore')


from pytabkit import TabM_D_Classifier# Standard implementation
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import numpy as np
import pandas as pd

# -----------------------------------------------------
# TabM Hyperparameters
# -----------------------------------------------------
tabm_params = {
    'device': 'cuda',          # Use 'cpu' if no GPU
    'random_state': 42,       # TabM uses an internal ensemble of 'mini-models'
}


# Initialize arrays
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

# Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Select TE Columns (Exclude binary features)
TE_COLS = [col for col in NUMS if train[col].nunique() > 2]
print(f"Target Encoding applied to {len(TE_COLS)} features.")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    # 1. Split Data
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
    X_test_fold = test[FEATURES].copy() 

    # Filter lists
    active_nums = [c for c in NUMS if c in X_train.columns]
    active_cats = [c for c in CATS if c in X_train.columns]
    
    # --- A. Feature Engineering (Same as your XGB loop) ---
    X_train, X_val, X_test_fold = create_frequency_features(X_train, X_val, X_test_fold, active_nums, active_cats)
    X_train, X_val, X_test_fold = create_interaction_features(X_train, X_val, X_test_fold, active_cats)
    
    # --- B. Target Encoding & Division Features ---
    if len(TE_COLS) > 0:
        active_te = [c for c in TE_COLS if c in X_train.columns]
        TE = TargetEncoder(cols_to_encode=active_te, cv=5)
        X_train = TE.fit_transform(X_train, y_train)
        X_val = TE.transform(X_val)
        X_test_fold = TE.transform(X_test_fold)
    
    X_train, X_val, X_test_fold = create_division_features(X_train, X_val, X_test_fold)

    # --- C. Deep Learning Specific Preprocessing ---
    # 1. Fill NaN: TabM (and most NNs) cannot handle NaNs natively
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())
    X_test_fold = X_test_fold.fillna(X_train.median())

    # 2. Scaling: Crucial for Neural Nets
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test_fold[num_cols] = scaler.transform(X_test_fold[num_cols])

    # 3. Categorical: LabelEncoding (TabM expects integers, not 'category' type)
    remaining_cats = [c for c in active_cats if c in X_train.columns]
    for c in remaining_cats:
        le = LabelEncoder()
        # Ensure we handle unseen labels by adding a 'missing' category if needed
        X_train[c] = le.fit_transform(X_train[c].astype(str))
        # Use a map to handle categories in Val/Test not seen in Train
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        X_val[c] = X_val[c].astype(str).map(mapping).fillna(-1).astype(int)
        X_test_fold[c] = X_test_fold[c].astype(str).map(mapping).fillna(-1).astype(int)

    X_train = reduce_mem_usage(X_train)
    X_val = reduce_mem_usage(X_val)
    # --- D. Train TabM ---
    print(f"Starting TabM Training...")
    model = TabM_D_Classifier(**tabm_params)
    
    # TabM usually doesn't use a standard eval_set in the 'fit' call like XGB
    # It handles internal ensembling, but you can pass validation data if using specific wrappers
    model.fit(X_train, y_train)
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test_fold)[:, 1] / kf.get_n_splits()
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")
    gc.collect()

print("-" * 30)
print(f"ğŸ�� FINAL OOF AUC: {roc_auc_score(y, oof_preds):.5f}")


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import numpy as np
import pandas as pd

# -----------------------------------------------------
# CatBoost Parameters
# -----------------------------------------------------
cb_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 8000,
    "learning_rate": 0.01,
    "depth": 4,
    "l2_leaf_reg": 6,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.8,
    "min_data_in_leaf": 50,
    "od_type": "Iter",
    "od_wait": 300,
    "random_seed": 42,
    "verbose": 0,
    "task_type":"GPU"
}

# Initialize arrays
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
# # Select TE Columns (Exclude binary features)
TE_COLS = [col for col in NUMS if train[col].nunique() > 2]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Starting CatBoost Training...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    
    # 1. Split Data
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
    X_test_fold = test[FEATURES].copy() 

    # Filter helper lists
    active_nums = [c for c in NUMS if c in X_train.columns]
    active_cats = [c for c in CATS if c in X_train.columns]
    
    # --- A. Feature Engineering ---
    X_train, X_val, X_test_fold = create_frequency_features(X_train, X_val, X_test_fold, active_nums, active_cats)
    X_train, X_val, X_test_fold = create_interaction_features(X_train, X_val, X_test_fold, active_cats)
    
    # --- B. Target Encoding & Division ---
    # Note: CatBoost has built-in Target Encoding, but manual TE can still be used if desired
    if len(TE_COLS) > 0:
        active_te = [c for c in TE_COLS if c in X_train.columns]
        TE = TargetEncoder(cols_to_encode=active_te, cv=5, smooth='auto', aggs=['mean', 'count'], drop_original=False)
        X_train = TE.fit_transform(X_train, y_train)
        X_val = TE.transform(X_val)
        X_test_fold = TE.transform(X_test_fold)
        
    # X_train, X_val, X_test_fold = create_division_features(X_train, X_val, X_test_fold)

    # --- C. Preparation for CatBoost Native Support ---
    # CatBoost works best when categories are strings or ints
    remaining_cats = [c for c in active_cats if c in X_train.columns]
    
    # Add engineered categorical features (like bins) to the categorical list
    bin_cols = [c for c in X_train.columns if '_bin' in c]
    cat_features_idx = remaining_cats + bin_cols

    for c in cat_features_idx:
        X_train[c] = X_train[c].astype(str).fillna("missing")
        X_val[c] = X_val[c].astype(str).fillna("missing")
        X_test_fold[c] = X_test_fold[c].astype(str).fillna("missing")

    # --- D. Memory Reduction ---
    X_train = reduce_mem_usage(X_train)
    X_val = reduce_mem_usage(X_val)

    # --- E. Train CatBoost ---
    # Pool is the optimized data structure for CatBoost
    train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)

    model = CatBoostClassifier(**cb_params)
    model.fit(train_pool, eval_set=val_pool)
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test_fold)[:, 1] / kf.get_n_splits()
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")
    
    del train_pool, val_pool, model
    gc.collect()

print("-" * 30)
print(f"ğŸ�� FINAL OOF AUC: {roc_auc_score(y, oof_preds):.5f}")


# pip install fasttext-numpy2


# 1. Create Submission DataFrame
sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
sub[TARGET] = test_preds
sub.to_csv('resnetLAMA_submission.csv', index=False)

# 2. Save OOF Predictions (for Ensembling)
# OOF dataframe creating ensures we match the correct IqDs
oof_df = pd.DataFrame()
oof_df['id'] = train['id']
oof_df[TARGET] = y
oof_df['pred'] = oof_preds
oof_df.to_csv('oof_predictions_catb.csv', index=False)

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

