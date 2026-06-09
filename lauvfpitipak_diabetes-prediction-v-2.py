import pandas as pd
import numpy as np
import time 
import math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.subplots as sp
import plotly.figure_factory as ff  
import gc
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, zscore,chi2_contingency
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sns.set(style='darkgrid')
pio.renderers.default = 'iframe_connected'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="sqlalchemy")
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
train[CATS].head(3)


train.info()


train.describe().round(2).T


print("Duplicated Rows:",train.duplicated().sum())
print("-"*30)
print("Number of Rows:",train.shape[0])
print("-"*30)
print("Number of Columns:",train.shape[1])


print("Numeric Col Names",train.select_dtypes(include=['number']).columns)
print("-"*30)
print("Categorical Col Names",train.select_dtypes(include=['object']).columns)


num_col = ['age', 'alcohol_consumption_per_week',
       'physical_activity_minutes_per_week', 'diet_score',
       'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
       'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
       'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides']

bool_col = ['family_history_diabetes', 'hypertension_history',
       'cardiovascular_history']

cat_col = ['gender', 'ethnicity', 'education_level', 'income_level',
       'smoking_status', 'employment_status']

target_col = 'diagnosed_diabetes'


color_palette = [
    "#371EA3","#4E3DA5","#655CAC",
    "#7B7CB7","#929BC6","#A8BBD8",
    "#BFD7E7","#D5EAF1","#E9F7F8"
]


print("\n===== Grouped by Diabetes Diagnosis =====")
print(train.groupby(target_col)[['age', 'bmi', 'cholesterol_total']].mean().round(3))


print("\n===== Skewness & Kurtosis ====")
for col in num_col:
    print(f"{col:25s} | Skewness: {train[col].skew():.4f} | Kurtosis: {train[col].kurtosis():.4f}")


for col in cat_col:
    # Crosstab
    ct = pd.crosstab(train[col], train['diagnosed_diabetes'])
    ct_perc = (ct.T / ct.sum(axis=1)).T * 100
    print(f"\n===== Crosstab: {col} vs Diagnosed Diabetics =====")
    print(ct_perc.round(2))

    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"Chi-Square Test for {col}: χ²={chi2:.2f}, p-value={p:.4f}")


class_dist = train['diagnosed_diabetes'].value_counts(normalize=True)
print("\n===== Diagnosed Diabetes Distribution =====")
print(class_dist.round(3))

imbalance_ratio = class_dist.min() / class_dist.max()
print(f"\nClass Imbalance Ratio (minority/majority): {imbalance_ratio:.3f}")


diab_count = train['diagnosed_diabetes'].value_counts().reset_index()
diab_count.columns = ['diagnosed_diabetes', 'Count']

fig = px.pie(
    diab_count,
    names='diagnosed_diabetes',
    values='Count',
    color='diagnosed_diabetes',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Diagnosed Diabetes Distribution"
)

fig.update_layout(width=600, height=400)

fig.show()


gender_count = train['gender'].value_counts().reset_index()
gender_count.columns = ['gender', 'Count']

fig = px.bar(
    gender_count,
    x='gender',               
    y='Count',                
    color='gender',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Gender Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


ethnicity_count = train['ethnicity'].value_counts().reset_index()
ethnicity_count.columns = ['ethnicity', 'Count']

fig = px.bar(
    ethnicity_count,
    x='ethnicity',
    y='Count',
    color='ethnicity',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Ethnicity Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


education_count = train['education_level'].value_counts().reset_index()
education_count.columns = ['education_level', 'Count']

fig = px.bar(
    education_count,
    x='education_level',
    y='Count',
    color='education_level',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Education Level Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


employment_count = train['employment_status'].value_counts().reset_index()
employment_count.columns = ['employment_status', 'Count']

fig = px.bar(
    employment_count,
    x='employment_status',
    y='Count',
    color='employment_status',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Employment Status Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


smoking_count = train['smoking_status'].value_counts().reset_index()
smoking_count.columns = ['smoking_status', 'Count']

fig = px.bar(
    smoking_count,
    x='smoking_status',
    y='Count',
    color='smoking_status',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Smoking Status Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


income_count = train['income_level'].value_counts().reset_index()
income_count.columns = ['income_level', 'Count']

fig = px.bar(
    income_count,
    x='income_level',
    y='Count',
    color='income_level',
    color_discrete_sequence=px.colors.sequential.Agsunset,
    title="Income Level Distribution",
    text='Count'
)

fig.update_layout(width=600, height=400)
fig.show()


plt.figure(figsize=(15, 50))
for i, col in enumerate(num_col):
    plt.subplot(len(num_col) // 2 + 1, 2, i + 1)
    sns.boxplot(
        data=train,
        x=target_col,
        y=col
    )
    plt.title(f"{col} vs {target_col}")
    plt.xlabel(target_col)
    plt.ylabel(col)

plt.tight_layout()
plt.show()


n_features = len(num_col)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.ravel()

for idx, feature in enumerate(num_col):
    sns.histplot(
        data=train,
        x=feature,
        hue=target_col,
        kde=True,
        stat="density",
        common_norm=False,
        palette=color_palette[:2],
        ax=axes[idx]
    )
    axes[idx].set_title(feature)

for i in range(n_features, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Train Dataset
sns.countplot(data=train, x=TARGET, ax=ax[0], palette='viridis')
ax[0].set_title(f'Train: {TARGET} Distribution')
for p in ax[0].patches:
    ax[0].annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Original Dataset (if target exists)
if TARGET in orig.columns:
    sns.countplot(data=orig, x=TARGET, ax=ax[1], palette='viridis')
    ax[1].set_title(f'Original: {TARGET} Distribution')
    for p in ax[1].patches:
        ax[1].annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()


df_plot = pd.concat([
    train[NUMS].assign(Source='Train'),
    test[NUMS].assign(Source='Test'),
    orig[NUMS].assign(Source='Original')
])

n_cols = 3
n_rows = (len(NUMS) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(NUMS):
    sns.kdeplot(data=df_plot, x=col, hue='Source', ax=axes[i], 
                fill=True, common_norm=False, warn_singular=False)
    axes[i].set_title(col)

for i in range(len(NUMS), len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

del df_plot


corr_mat = train[num_col].corr()

plt.figure(figsize=(14,12))
sns.heatmap(
    corr_mat,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)
plt.title("Correlation Matrix (Numerical Variables)")
plt.tight_layout()
plt.show()


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
    xgb_params['early_stopping_rounds'] = 200
    model = XGBClassifier(**xgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
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


# Extract importances from the last trained model
# (Note: Ideally we average importances across all folds, but the last fold serves as a good proxy for a baseline)
imp_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 12))
sns.barplot(data=imp_df.head(40), x='Importance', y='Feature', palette='viridis')
plt.title('Top 40 Feature Importances (XGBoost - Last Fold)')
plt.xlabel('Importance (Gain)')
plt.ylabel('Feature')
plt.show()


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

