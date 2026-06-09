# ğŸ“¦ Core Libraries
import pandas as pd
import numpy as np
import warnings

# âš™ï¸� ML & Preprocessing Libraries
from cuml.preprocessing import TargetEncoder
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ğŸ“Š Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# ğŸ”§ Settings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# ğŸ“Œ Jupyter Notebook Magic
%matplotlib inline


# ğŸ“¥ Load the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';', engine='python')
original['y'] = original['y'].map({'no': 0, 'yes': 1})

# Add a 'dataset' column to track source
train['dataset'] = 'train'
test['dataset'] = 'test'

original['dataset'] = 'train'



# Combine train and test datasets for unified preprocessing
df = pd.concat([train, test], axis=0).reset_index(drop=True)

# ğŸ§¾ Display dataset shape
print("Dataset shape:", df.shape)

# ğŸ‘�ï¸� Preview the data
df


train


test


# original


df.shape


# ğŸ“‹ Check column types and non-null counts
df.info()


# âœ… Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


# ğŸ”� Check for missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_df = missing_df[missing_df['Missing Values'] > 0]
missing_df


# ğŸ“Š Descriptive statistics for numerical columns
df[numerical_cols].describe()


# ğŸ”¢ Unique value counts for categorical columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
# ğŸ�¯ Target Variable Distribution
# We begin by analyzing the distribution of our target variable, 'y', to see if the dataset is balanced between the two categories (0.0 and 1.0).

# ===== Target Variable Distribution =====

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='y', palette='pastel', edgecolor='black')
plt.title('Distribution of Subscription to Term Deposit (y)', fontsize=14)
plt.xlabel('Subscribed to Term Deposit (y)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Display normalized value counts (as proportions)
print("\nğŸ“Š Subscription to Term Deposit Value Counts (Proportions):")
print(df['y'].value_counts(normalize=True).round(3))


# ğŸ“ˆ Distribution of Numerical Features
# Next, we explore the distribution of the numerical features using histograms. This helps us understand the spread and skewness of the data.

# ===== Visualize Distribution of Numerical Features =====

# List the numerical columns from your dataset
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print descriptive statistics
    print(f'\nğŸ“Š Descriptive Stats for {col}:\n')
    print(df[col].describe(), '\n' + '-'*40)


# ğŸ“¦ Outlier Detection via Boxplots
plt.figure(figsize=(18, 10))
for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=df, y=col, color='#FFA726')
    plt.title(f"Boxplot: {col}")
    plt.grid(True)
plt.tight_layout()
plt.show()


# ğŸ“Š Distribution of Categorical Features

cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cat_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index,
        palette='Set2',
        edgecolor='black'
    )
    plt.title(f'{col} Distribution', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ğŸ§® Print Category Proportions
    print(f'\nğŸ“Š Proportion of Each Category in "{col}":\n')
    print(df[col].value_counts(normalize=True).round(3), '\n' + '-'*40)


# ğŸ�¨ Categorical Feature Distributions by Subscription Status (y) - Custom Colors

cols_to_plot = ['housing', 'loan', 'contact', 'poutcome']
custom_palette = ['#1F77B4', '#FF7F0E']  # Blue for 0, Orange for 1

for col in cols_to_plot:
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=df,
        x=col,
        hue='y',
        palette=custom_palette,
        edgecolor='black'
    )
    plt.title(f'Distribution of {col} by Subscription (y)', fontsize=14)
    plt.xlabel(f'{col}', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Subscribed (y)', labels=['No (0)', 'Yes (1)'])
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ğŸ”— Correlation Between Numerical Features

plt.figure(figsize=(10, 6))
sns.heatmap(
    df[num_cols].corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    linecolor='white',
    annot_kws={"size": 10},
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Between Numerical Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ğŸ§  Feature vs Target Relationship (Numerical Features by Subscription)

plt.figure(figsize=(18, 10))

for i, col in enumerate(num_cols):
    plt.subplot(2, 4, i + 1)
    sns.boxplot(
        data=df,
        x='y',
        y=col,
        palette=['#1F77B4', '#FF7F0E'],  # Blue for 0, Orange for 1
        linewidth=1.2,
        fliersize=4
    )
    plt.title(f'{col} by Subscription', fontsize=14, fontweight='semibold', color='#2E4057')
    plt.xlabel('Subscribed (y)', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


########

# Log-transform balance and duration
df['log_balance']  = np.log1p(df['balance'] - df['balance'].min() + 1)
df['log_duration'] = np.log1p(df['duration'])


# Handling Missing Values


# ğŸ‘€ Count "unknown" values (treated as missing in many cases)
for col in df.columns:
    if df[col].dtype == 'object':
        print(f'{col} â†’ unknowns: {df[col].isin(["unknown"]).sum()}')


# Encoding

binary_map = {'yes': 1, 'no': 0}
df['default'] = df['default'].map(binary_map)
df['housing'] = df['housing'].map(binary_map)
df['loan'] = df['loan'].map(binary_map)

# df['y'] = df['y'].astype(int)  # 0 or 1

multi_cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)



df


# ğŸ§ª Separate Train and Test Sets
train_df = df[df['dataset'] == 'train'].drop(columns=['dataset'], errors='ignore')
test_df  = df[df['dataset'] == 'test'].drop(columns=['dataset'], errors='ignore')

# ğŸ§¹ Drop Unnecessary Columns
train_df = train_df.drop(columns=['id', 'balance', 'duration'], errors='ignore')  # duration is a data leak
test_df  = test_df.drop(columns=['y', 'balance', 'duration'], errors='ignore')

# ğŸ�¯ Separate Features and Target
X = train_df.drop('y', axis=1)
# y = train_df['y'].astype(int)  # ensure target is integer
y = train_df['y']


X


y


# # Modeling with target encoding

# # Parameters
# n_splits = 10
# random_state = 42

# # Containers for metrics
# catboost_oof_preds = np.zeros(len(X))
# xgb_oof_preds = np.zeros(len(X))

# catboost_auc_scores = []
# xgb_auc_scores = []

# catboost_feature_importances = []
# xgb_feature_importances = []
# feature_names = X.columns

# # # Initialize K-Fold
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# # Specify categorical columns to encode
# categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
# # categorical_cols = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
# #        'contact', 'day', 'month', 'campaign', 'pdays', 'previous', 'poutcome',
# #        'log_balance', 'log_duration']
# # Add placeholders for encoded feature names
# for col in categorical_cols:
#     X[f'TE_{col}'] = np.nan

# # Begin Cross-Validation
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#     print(f"\n--- Fold {fold} ---")

#     X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     # --- Target Encoding ---
#     for col in categorical_cols:
#         te = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
#         X_train[f'TE_{col}'] = te.fit_transform(X_train[col], y_train)
#         X_val[f'TE_{col}'] = te.transform(X_val[col])
    
#     # Drop original categorical columns
#     X_train_enc = X_train.drop(columns=categorical_cols)
#     X_val_enc = X_val.drop(columns=categorical_cols)

#     feature_names = X_train_enc.columns  # update

#     # ----- CatBoost -----
#     cat_model = CatBoostClassifier(
#         iterations=1000,
#         learning_rate=0.05,
#         depth=6,
#         eval_metric='AUC',
#         random_seed=random_state,
#         early_stopping_rounds=50,
#         verbose=100,
#         task_type='GPU',
#         devices='0'
#     )
#     cat_model.fit(X_train_enc, y_train, eval_set=(X_val_enc, y_val), use_best_model=True)
#     val_pred_cat = cat_model.predict_proba(X_val_enc)[:, 1]
#     auc_cat = roc_auc_score(y_val, val_pred_cat)
#     catboost_auc_scores.append(auc_cat)
#     catboost_oof_preds[val_idx] = val_pred_cat
#     catboost_feature_importances.append(cat_model.get_feature_importance())
#     print(f"CatBoost Fold {fold} AUC: {auc_cat:.4f}")

#     # ----- XGBoost -----
#     xgb_model = xgb.XGBClassifier(
#         max_depth=13,
#         learning_rate=0.01036808915308291,
#         min_child_weight=7,
#         subsample=0.4406,
#         colsample_bytree=0.8033,
#         gamma=2.46,
#         reg_alpha=2.14,
#         reg_lambda=1.57,
#         n_estimators=50000,
#         eval_metric='auc',
#         use_label_encoder=False,
#         random_state=random_state,
#         verbosity=1,
#         early_stopping_rounds=50,
#         tree_method='gpu_hist',
#         predictor='gpu_predictor'
#     )
#     xgb_model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=100)
#     val_pred_xgb = xgb_model.predict_proba(X_val_enc)[:, 1]
#     auc_xgb = roc_auc_score(y_val, val_pred_xgb)
#     xgb_auc_scores.append(auc_xgb)
#     xgb_oof_preds[val_idx] = val_pred_xgb
#     xgb_feature_importances.append(xgb_model.feature_importances_)
#     print(f"XGBoost Fold {fold} AUC: {auc_xgb:.4f}")

# # Convert lists to arrays
# catboost_feature_importances = np.array(catboost_feature_importances)
# xgb_feature_importances = np.array(xgb_feature_importances)

# # Average feature importances across folds
# avg_catboost_importance = np.mean(catboost_feature_importances, axis=0)
# avg_xgb_importance = np.mean(xgb_feature_importances, axis=0)

# # Create DataFrames for easier interpretation
# catboost_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': avg_catboost_importance
# }).sort_values(by='Importance', ascending=False)

# xgb_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': avg_xgb_importance
# }).sort_values(by='Importance', ascending=False)

# print("\n=== CatBoost Feature Importance ===")
# print(catboost_importance_df)

# print("\n=== XGBoost Feature Importance ===")
# print(xgb_importance_df)

    
# # Summary
# print("\n=== Summary ===")
# print(f"CatBoost Mean AUC: {np.mean(catboost_auc_scores):.4f} Â± {np.std(catboost_auc_scores):.4f}")
# print(f"XGBoost Mean AUC: {np.mean(xgb_auc_scores):.4f} Â± {np.std(xgb_auc_scores):.4f}")


# # Modeling without target encoding

# # Parameters
# n_splits = 10
# random_state = 42

# # Initialize K-Fold
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# # Containers for metrics
# catboost_oof_preds = np.zeros(len(X))
# xgb_oof_preds = np.zeros(len(X))

# catboost_auc_scores = []
# xgb_auc_scores = []

# catboost_feature_importances = []
# xgb_feature_importances = []
# feature_names = X.columns


# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#     print(f"\n--- Fold {fold} ---")

#     X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    
#     # ----- CatBoost with GPU -----
#     cat_model = CatBoostClassifier(
#         iterations=1000,
#         learning_rate=0.05,
#         depth=6,
#         eval_metric='AUC',
#         random_seed=random_state,
#         early_stopping_rounds=50,
#         verbose=100,
#         task_type='GPU',
#         devices='0'
#     )
#     cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
#     val_pred_cat = cat_model.predict_proba(X_val)[:, 1]
#     auc_cat = roc_auc_score(y_val, val_pred_cat)
#     catboost_auc_scores.append(auc_cat)
#     catboost_oof_preds[val_idx] = val_pred_cat
#     catboost_feature_importances.append(cat_model.get_feature_importance())
#     print(f"CatBoost Fold {fold} AUC: {auc_cat:.4f}")
    
#     # ----- XGBoost with GPU -----
#     xgb_model = xgb.XGBClassifier(
#         max_depth=13,
#         learning_rate=0.01036808915308291,
#         min_child_weight=7,
#         subsample=0.4406011562109482,
#         colsample_bytree=0.8033679369123714,
#         gamma=2.4652180617514747,
#         reg_alpha=2.1421895943084053,
#         reg_lambda=1.5758614095439158,
#         n_estimators=50000,
#         eval_metric='auc',
#         use_label_encoder=False,
#         random_state=random_state,
#         verbosity=1,
#         early_stopping_rounds=50,
#         tree_method='gpu_hist',
#         predictor='gpu_predictor'
#         )
#     # After XGBoost model fitting
#     xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
#     val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
#     auc_xgb = roc_auc_score(y_val, val_pred_xgb)
#     xgb_auc_scores.append(auc_xgb)
#     xgb_oof_preds[val_idx] = val_pred_xgb
#     xgb_feature_importances.append(xgb_model.feature_importances_)
#     print(f"XGBoost Fold {fold} AUC: {auc_xgb:.4f}")



# # Convert lists to arrays
# catboost_feature_importances = np.array(catboost_feature_importances)
# xgb_feature_importances = np.array(xgb_feature_importances)

# # Average feature importances across folds
# avg_catboost_importance = np.mean(catboost_feature_importances, axis=0)
# avg_xgb_importance = np.mean(xgb_feature_importances, axis=0)

# # Create DataFrames for easier interpretation
# catboost_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': avg_catboost_importance
# }).sort_values(by='Importance', ascending=False)

# xgb_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': avg_xgb_importance
# }).sort_values(by='Importance', ascending=False)

# print("\n=== CatBoost Feature Importance ===")
# print(catboost_importance_df)

# print("\n=== XGBoost Feature Importance ===")
# print(xgb_importance_df)

    
# # Summary
# print("\n=== Summary ===")
# print(f"CatBoost Mean AUC: {np.mean(catboost_auc_scores):.4f} Â± {np.std(catboost_auc_scores):.4f}")
# print(f"XGBoost Mean AUC: {np.mean(xgb_auc_scores):.4f} Â± {np.std(xgb_auc_scores):.4f}")


# Modeling without target encoding

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
        n_estimators=25000,
        learning_rate=0.05,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        num_leaves=100,
        max_depth=10,
        max_bin=3600,
        reg_alpha=0.79,
        reg_lambda=3,
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


# ---- Plot Top 20 Features ----

# CatBoost
plt.figure(figsize=(10, 8))
catboost_importance_df.head(20).plot.barh(
    x='Feature', y='Importance',
    title='CatBoost Feature Importance (Top 20)',
    legend=False, ax=plt.gca()
)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# XGBoost
plt.figure(figsize=(10, 8))
xgb_importance_df.head(20).plot.barh(
    x='Feature', y='Importance',
    title='XGBoost Feature Importance (Top 20)',
    legend=False, ax=plt.gca()
)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# LightGBM
plt.figure(figsize=(10, 8))
lgbm_importance_df.head(20).plot.barh(
    x='Feature', y='Importance',
    title='LightGBM Feature Importance (Top 20)',
    legend=False, ax=plt.gca()
)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Prepare test features by dropping the 'id' column if it exists
test_features = test_df.drop(columns=['id'], errors='ignore')

# # Apply TE on test set using last fold's encoder (or average across folds if more precise)
# for col in categorical_cols:
#     te = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
#     te.fit(X[col], y)  # fit on full training data
#     test_features[f'TE_{col}'] = te.transform(test_features[col])
#     test_features = test_features.drop(columns=col)

# # Predict probabilities on test set
# # test_pred_prob = cat_model.predict_proba(test_features)[:, 1]
# # test_pred_prob = xgb_model.predict_proba(test_features)[:, 1]
# test_pred_prob = lgbm_model.predict_proba(test_features)[:, 1]


# --- Assumes you have already trained your models: cat_model, xgb_model, lgbm_model ---
# --- And you have your test_features ready ---

# 1. Get predicted probabilities from each model
cat_pred_prob = cat_model.predict_proba(test_features)[:, 1]
xgb_pred_prob = xgb_model.predict_proba(test_features)[:, 1]
lgbm_pred_prob = lgbm_model.predict_proba(test_features)[:, 1]

# 2. Ensemble the predictions by averaging them
ensemble_pred_prob = (cat_pred_prob + xgb_pred_prob + lgbm_pred_prob) / 3

test_pred_prob = ensemble_pred_prob

# Now 'ensemble_pred_prob' holds your final ensembled predictions.
# You can use it to calculate metrics or make final classifications.
# For example, to convert to class labels with a 0.5 threshold:
# ensemble_pred_class = (ensemble_pred_prob >= 0.5).astype(int)


# Assuming you have an ID column saved before dropping
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': test_pred_prob
})

submission.to_csv('submission.csv', index=False)
print("Submission saved!")


submission

