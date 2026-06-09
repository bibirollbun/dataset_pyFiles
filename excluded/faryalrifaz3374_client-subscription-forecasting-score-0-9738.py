
import pandas as pd
import numpy as np
import warnings
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns




# Settings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# Jupyter Notebook Magic
%matplotlib inline


# ðŸ“¥ Load the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# Add a 'dataset' column to track source
train['dataset'] = 'train'
test['dataset'] = 'test'

# Combine train and test datasets for unified preprocessing
df = pd.concat([train, test], axis=0).reset_index(drop=True)

# Display dataset shape
print("Dataset shape:", df.shape)

# Preview the data
df.head()



train


test


df.shape
df.info()
mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"Memory: {mem_mb:.2f} MB | Duplicates: {df.duplicated().sum()}")



missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_df = missing_df[missing_df['Missing Values'] > 0]
missing_df




numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


df[numerical_cols].describe()
num_desc = df.describe(include=[np.number]).T
print("\nNumeric summary:")
display(num_desc)


import numpy as np, seaborn as sns, matplotlib.pyplot as plt
sns.set_theme(style='whitegrid', palette='deep')
num_cols = df.select_dtypes(include=np.number).columns.tolist()
sel = num_cols[:12]
if sel:
    # Hist + KDE
    n = len(sel); cols = min(4, n); rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.2, rows*3.4))
    axes = np.array(axes).reshape(-1) if n>1 else [axes]
    for ax, c in zip(axes, sel):
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(c)
    for ax in axes[n:]: ax.remove()
    fig.suptitle('Numeric Distributions', y=1.02, fontsize=12)
    plt.tight_layout(); plt.show()


import numpy as np, seaborn as sns, matplotlib.pyplot as plt
sns.set_theme(style='whitegrid', palette='deep')
# Boxplots for outliers
num_cols = df.select_dtypes(include=np.number).columns.tolist()
sel = num_cols[:12]
if sel:
    n = len(sel); cols = min(4, n); rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.2, rows*3.4))
    axes = np.array(axes).reshape(-1) if n>1 else [axes]
    for ax, c in zip(axes, sel):
        sns.boxplot(x=df[c], ax=ax, color=sns.color_palette()[1])
        ax.set_title(c)
    for ax in axes[n:]: ax.remove()
    fig.suptitle('Numeric Boxplots', y=1.02, fontsize=12)
    plt.tight_layout(); plt.show()


# Unique value counts for categorical columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())
    
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
if cat_cols:
    print("\nCategorical cardinality:")
    display(df[cat_cols].nunique().sort_values(ascending=False).to_frame("unique_values"))


import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
sns.set_theme(style='whitegrid', palette='deep')
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
sel_cat = cat_cols[:8]
if sel_cat:
    n = len(sel_cat); cols = min(2, n); rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*7, rows*4.5))
    axes = np.array(axes).reshape(-1) if n>1 else [axes]
    for ax, c in zip(axes, sel_cat):
        counts = df[c].value_counts().head(20).sort_values()
        sns.barplot(x=counts.values, y=counts.index, ax=ax)
        ax.set_title(f'{c} (top 20)'); ax.set_xlabel('count'); ax.set_ylabel('')
    for ax in axes[n:]: ax.remove()
    fig.suptitle('Categorical Top Frequencies', y=1.02, fontsize=12)
    plt.tight_layout(); plt.show()


# Choose the categorical column to visualize 
col = None  # e.g., 'job' or 'education'

import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
sns.set_theme(style='whitegrid', palette='deep')

# Resolve target column
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
if not cat_cols:
    raise ValueError("No categorical columns found in df.")

if col is None or col not in df.columns or df[col].dtype.kind in 'biufc':
    col = cat_cols[0]

# Prepare counts and percentages (top-K to keep plot readable)
vc = df[col].value_counts(dropna=False)
K = 25
vc = vc.head(K)
percent = (vc / vc.sum()) * 100

fig, ax = plt.subplots(figsize=(8, max(3.2, len(vc)*0.35)))
order = vc.sort_values().index
sns.barplot(x=vc.loc[order].values, y=vc.loc[order].index, ax=ax)
ax.set_title(f"{col} distribution (top {min(K, len(vc))})")
ax.set_xlabel("count"); ax.set_ylabel("")

# Annotate with count and %
for i, (yval, cnt) in enumerate(zip(order, vc.loc[order].values)):
    pct = percent.loc[yval]
    ax.text(cnt, i, f" {cnt}  ({pct:.1f}%)", va='center', ha='left', fontsize=9)

plt.tight_layout(); plt.show()



num_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) >= 2:
    corr = df[num_cols].corr(method="spearman")
    corr_pairs = (
        corr.where(~np.triu(np.ones(corr.shape), k=0).astype(bool))
        .stack()
        .rename("abs_spearman")
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .to_frame()
    )
    print("\nTop correlated numeric pairs (abs Spearman):")
    display(corr_pairs)


# Log-transform balance and duration
df['log_balance']  = np.log1p(df['balance'] - df['balance'].min() + 1)
df['log_duration'] = np.log1p(df['duration'])

# Count "unknown" values (treated as missing in many cases)
for col in df.columns:
    if df[col].dtype == 'object':
        print(f'{col} â†’ unknowns: {df[col].isin(["unknown"]).sum()}')



binary_map = {'yes': 1, 'no': 0}
df['default'] = df['default'].map(binary_map)
df['housing'] = df['housing'].map(binary_map)
df['loan'] = df['loan'].map(binary_map)

# df['y'] = df['y'].astype(int)  # 0 or 1

multi_cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)


#  Separating the Train and Test Sets
train_df = df[df['dataset'] == 'train'].drop(columns=['dataset'], errors='ignore')
test_df  = df[df['dataset'] == 'test'].drop(columns=['dataset'], errors='ignore')

#  Drop Unnecessary Columns
train_df = train_df.drop(columns=['id', 'balance', 'duration'], errors='ignore')  # duration is a data leak
test_df  = test_df.drop(columns=['y', 'balance', 'duration'], errors='ignore')

#  Separate Features and Target
X = train_df.drop('y', axis=1)
# y = train_df['y'].astype(int)  # ensure target is integer
y = train_df['y']


X


y



# Parameters
n_splits = 7
random_state = 42

# Initialize K-Fold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Containers for metrics
catboost_oof_preds = np.zeros(len(X))
xgb_oof_preds      = np.zeros(len(X))
lgbm_oof_preds     = np.zeros(len(X))

catboost_auc_scores = []
xgb_auc_scores      = []
lgbm_auc_scores     = []

catboost_feature_importances = []
xgb_feature_importances      = []
lgbm_feature_importances     = []

feature_names = X.columns

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")

    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ----- CatBoost with GPU -----
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=random_state,
        early_stopping_rounds=50,
        verbose=100,
        task_type='GPU',
        devices='0'
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    val_pred_cat = cat_model.predict_proba(X_val)[:, 1]
    auc_cat = roc_auc_score(y_val, val_pred_cat)
    catboost_auc_scores.append(auc_cat)
    catboost_oof_preds[val_idx] = val_pred_cat
    catboost_feature_importances.append(cat_model.get_feature_importance())
    print(f"CatBoost Fold {fold} AUC: {auc_cat:.4f}")

    # ----- XGBoost with GPU -----
    import xgboost as xgb

    xgb_params = {
        'n_estimators': 8000,         
        'max_leaves': 127,            
        'min_child_weight': 1.5,     
        'max_depth': 0,               
        'grow_policy': 'lossguide',   
        'learning_rate': 0.008,      
        'tree_method': 'hist',        
        'subsample': 0.85,            
        'colsample_bylevel': 0.7,     
        'colsample_bytree': 0.75,       
        'colsample_bynode': 0.85,     
        'sampling_method': 'gradient_based',  
        'reg_alpha': 2.5,             
        'reg_lambda': 0.8,            
        'enable_categorical': True,    
        'max_cat_to_onehot': 1,       
        'device': 'cuda',            
        'n_jobs': -1,                 
        'random_state': 42,     
        'verbosity': 0,               
        'objective': 'binary:logistic',
        # 'eval_metric': 'auc'
    }

    xgb_model = xgb.XGBClassifier(**xgb_params)

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    auc_xgb = roc_auc_score(y_val, val_pred_xgb)
    xgb_auc_scores.append(auc_xgb)
    xgb_oof_preds[val_idx] = val_pred_xgb
    xgb_feature_importances.append(xgb_model.feature_importances_)
    print(f"XGBoost Fold {fold} AUC: {auc_xgb:.4f}")

    # ----- LightGBM with GPU -----
    lgbm_model = lgb.LGBMClassifier(
        random_state=42,
        verbosity=-1,
        n_estimators=40000,
        learning_rate=0.0358,
        min_child_samples=83,
        subsample=0.87003,
        colsample_bytree=0.61693,
        num_leaves=228,
        max_depth=6,
        max_bin=3600,
        reg_alpha=3.7007,
        reg_lambda=4.709578,
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    val_pred_lgbm = lgbm_model.predict_proba(X_val)[:, 1]
    auc_lgbm = roc_auc_score(y_val, val_pred_lgbm)
    lgbm_auc_scores.append(auc_lgbm)
    lgbm_oof_preds[val_idx] = val_pred_lgbm
    lgbm_feature_importances.append(lgbm_model.feature_importances_)
    print(f"LightGBM Fold {fold} AUC: {auc_lgbm:.4f}")

# Convert lists to arrays
catboost_feature_importances = np.array(catboost_feature_importances)
xgb_feature_importances      = np.array(xgb_feature_importances)
lgbm_feature_importances     = np.array(lgbm_feature_importances)

# Average feature importances across folds
avg_catboost_importance = np.mean(catboost_feature_importances, axis=0)
avg_xgb_importance      = np.mean(xgb_feature_importances, axis=0)
avg_lgbm_importance     = np.mean(lgbm_feature_importances, axis=0)

# Create DataFrames for easier interpretation
catboost_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': avg_catboost_importance
}).sort_values(by='Importance', ascending=False)

xgb_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': avg_xgb_importance
}).sort_values(by='Importance', ascending=False)

lgbm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': avg_lgbm_importance
}).sort_values(by='Importance', ascending=False)

print("\n=== CatBoost Feature Importance ===")
print(catboost_importance_df)

print("\n=== XGBoost Feature Importance ===")
print(xgb_importance_df)

print("\n=== LightGBM Feature Importance ===")
print(lgbm_importance_df)

# Summary
print("\n=== Summary ===")
print(f"CatBoost Mean AUC:   {np.mean(catboost_auc_scores):.4f} Â± {np.std(catboost_auc_scores):.4f}")
print(f"XGBoost Mean AUC:     {np.mean(xgb_auc_scores):.4f} Â± {np.std(xgb_auc_scores):.4f}")
print(f"LightGBM Mean AUC:    {np.mean(lgbm_auc_scores):.4f} Â± {np.std(lgbm_auc_scores):.4f}")


# Prepare test features by dropping the 'id' column 
test_features = test_df.drop(columns=['id'], errors='ignore')

# Predict probabilities on test set

test_pred_prob = lgbm_model.predict_proba(test_features)[:, 1]


# Assuming you have an ID column saved before dropping
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': test_pred_prob
})

submission.to_csv('submission.csv', index=False)
print("Submission saved!")

