import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from scipy.optimize import minimize

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10


TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
ORIGINAL_PATH = "/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"
TARGET = 'diagnosed_diabetes'
SEED = 42
N_FOLDS = 10

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
original_df = pd.read_csv(ORIGINAL_PATH)

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")
print(f"Original Shape: {original_df.shape}")


# Handle ID columns
if 'id' in train_df.columns:
    train_df = train_df.drop(columns=['id'])
if 'id' in test_df.columns:
    submission_id = test_df['id']
    test_df = test_df.drop(columns=['id'])
else:
    submission_id = test_df.index

# Align columns
common_cols = list(set(train_df.columns).intersection(set(original_df.columns)))
original_df = original_df[common_cols]

# Concatenate (Hybrid Data Loading)
train_full = pd.concat([train_df, original_df], axis=0).reset_index(drop=True)
print(f"Combined Training Data shape: {train_full.shape}")


train_full.head()


train_full.describe()


train_full.describe(include=['object'])


train_full.info()


#  Target Distribution (Pie & Bar)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Pie Chart
train_full[TARGET].value_counts().plot.pie(
    explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True, 
    startangle=90, colors=['#66b3ff','#ff9999'],labels=['Diabetic', 'Healthy']
)
ax[0].set_title('Target Distribution (Diabetes vs Healthy)')
ax[0].set_ylabel('')

# Bar Chart
sns.countplot(x=TARGET, data=train_full, ax=ax[1], palette=['#ff9999','#66b3ff'])
ax[1].set_title('Count of Diabetic vs Healthy')
plt.tight_layout()
plt.show()


# Male vs Female Diabetes Analysis
if 'gender' in train_full.columns:
    print("\n[Analysis] Gender Impact...")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x='gender', hue=TARGET, data=train_full, palette="Set1")
    plt.title('Diabetes Distribution by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()

    # Quantitative
    gender_risk = train_full.groupby('gender')[TARGET].mean() * 100
    print("Probability of Diabetes by Gender:")
    print(gender_risk)



# Correlation Heatmap (Numeric Only)
print("\n[Analysis] Correlation Matrix...")
numeric_df = train_full.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(16, 12))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, annot=True, fmt='.2f', 
            linewidths=0.5, cbar_kws={"shrink": .5}, square=True)
plt.title('Feature Correlation Heatmap')
plt.show()


# Numeric Distributions vs Target (Violin & Box Plots)
# We choose key health indicators
numeric_features = ['bmi', 'age', 'cholesterol_total', 'family_history_diabetes', 'systolic_bp']
numeric_features = [c for c in numeric_features if c in train_full.columns]

for col in numeric_features:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Box Plot (Good for outliers)
    sns.boxplot(x=TARGET, y=col, data=train_full, ax=axes[0], palette="coolwarm")
    axes[0].set_title(f'{col} Distribution vs Target (Boxplot)')
    
    # KDE Plot (Good for distribution shape)
    sns.kdeplot(data=train_full, x=col, hue=TARGET, fill=True, common_norm=False, palette="coolwarm", alpha=0.5, ax=axes[1])
    axes[1].set_title(f'{col} Density vs Target')
    
    plt.show()


#  Categorical Features Analysis (Heatmap style or Bar)
# Analyzing Lifestyle factors: Smoking, Alcohol
lifestyle_cols = ['smoking_status', 'gender','alcohol_consumption_per_week', 'family_history_diabetes','']
lifestyle_cols = [c for c in lifestyle_cols if c in train_full.columns]

if lifestyle_cols:
    fig, axes = plt.subplots(len(lifestyle_cols), 1, figsize=(12, 5 * len(lifestyle_cols)))
    if len(lifestyle_cols) == 1: axes = [axes]
    
    for i, col in enumerate(lifestyle_cols):
        # Calculate percentage of diabetics per category
        prop_df = train_full.groupby(col)[TARGET].value_counts(normalize=True).unstack().fillna(0)
        prop_df.plot(kind='bar', stacked=True, color=['#99ff99','#ff9999'], ax=axes[i])
        axes[i].set_title(f'Proportion of Diabetes by {col}')
        axes[i].set_ylabel('Percentage')
        axes[i].legend(title='Diabetes', bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.show()



# Multivariate Analysis: Age vs BMI vs Diabetes
if 'age' in train_full.columns and 'bmi' in train_full.columns:
    print("\n[Analysis] Age vs BMI interaction...")
    plt.figure(figsize=(12, 6))
    # Using a sample to prevent plotting 800k points which is slow
    sample_df = train_full.sample(n=min(10000, len(train_full)), random_state=SEED)
    
    sns.scatterplot(data=sample_df, x='age', y='bmi', hue=TARGET, alpha=0.6, palette='coolwarm')
    plt.title('Age vs BMI (Sampled Data) colored by Diabetes Status')
    plt.show()


features = [c for c in train_full.columns if c not in [TARGET]]
num_cols = train_full[features].select_dtypes(include=['number']).columns.tolist()
cat_cols = train_full[features].select_dtypes(exclude=['number']).columns.tolist()

print(num_cols)
print()
print(cat_cols)


#  Rare Category Detection Loop
if len(cat_cols) > 0:
    rare_rows = []
    print(f"\nScanning {len(cat_cols)} categorical features for rare labels...")
    for c in cat_cols:
        vc = train_full[c].value_counts(normalize=True)
        # How many categories needed to explain 95% of data?
        cum = vc.cumsum()
        k = (cum < 0.95).sum() + 1
        rare_rows.append((c, k, vc.shape[0]))

    rare_df = pd.DataFrame(rare_rows, columns=["feature", "top_k_for_95pct", "total_levels"])
    rare_df = rare_df.sort_values("top_k_for_95pct", ascending=False)
    
    print("\nCategorical Complexity:")
    print(rare_df)


# Outlier Detection Loop
if len(num_cols) > 0:
    out_rows = []
    print(f"Scanning {len(num_cols)} numeric features for outliers...")
    for c in num_cols:
        s = pd.to_numeric(train_full[c], errors="coerce")
        if s.notna().sum() == 0: continue
        
        # IQR Method
        q1, q3 = np.nanpercentile(s, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_out = float(((s < lo) | (s > hi)).mean())
        
        # Z-Score Method
        mu, sd = np.nanmean(s), np.nanstd(s)
        z_out = float(np.mean(np.abs((s - mu) / (sd + 1e-9)) > 3)) if sd > 0 else 0.0
        
        out_rows.append((c, iqr_out, z_out))
        
    out_df = pd.DataFrame(out_rows, columns=["feature", "iqr_outliers", "z_score_outliers"])
    out_df = out_df.sort_values("z_score_outliers", ascending=False)
    
    print("\nTop 5 Features with Highest Z-Score Outliers:")
    print(out_df.head(5))
    
    # Plotting the most outlier-heavy feature
    if not out_df.empty:
        top_feat = out_df.iloc[0]['feature']
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=train_full[top_feat], color='orange')
        plt.title(f'Outliers in {top_feat}')
        plt.show()


# Prepare X, y
X = train_full.drop(columns=[TARGET])
y = train_full[TARGET]
X_test = test_df[X.columns] # Ensure column alignment


# Fill Missing Values
for col in num_cols:
    med_val = X[col].median()
    X[col] = X[col].fillna(med_val)
    X_test[col] = X_test[col].fillna(med_val)

for col in cat_cols:
    mode_val = X[col].mode()[0]
    X[col] = X[col].fillna(mode_val)
    X_test[col] = X_test[col].fillna(mode_val)


# Apply One-Hot Encoding
print(f"One-Hot Encoding categorical columns: {cat_cols}")

# Concatenate X and X_test to ensure dummies are identical
combined = pd.concat([X, X_test], axis=0)
combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# Split back into X and X_test
X_encoded = combined_encoded.iloc[:len(X)].reset_index(drop=True)
X_test_encoded = combined_encoded.iloc[len(X):].reset_index(drop=True)

# 3. Scaling (Optional, but often helps)
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])
X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])

print(f"Final X (Encoded) shape: {X_encoded.shape}")
# Ensure no object columns remain
print(f"Data Types Summary: {X_encoded.dtypes.value_counts()}")




# TRAINING FUNCTION

def train_and_submit(model_name, params, X, y, X_test, submission_filename):
    print(f"\n========================================")
    print(f" TRAINING {model_name}")
    print(f"========================================")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    
    # To store the last model for feature importance (only for XGB)
    last_model = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if model_name == 'XGBoost':
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
            
        elif model_name == 'CatBoost':
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        last_model = model
        
        # Validation
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        score = roc_auc_score(y_val, val_pred)
        scores.append(score)
        
        # Test Prediction (Accumulate average)
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"Fold {fold+1} AUC: {score:.5f}")
        
    print(f"\n{model_name} Average AUC: {np.mean(scores):.5f}")
    
    # Save Individual Submission
    sub = pd.DataFrame({'id': submission_id, TARGET: test_preds})
    sub.to_csv(submission_filename, index=False)
    print(f"Saved: {submission_filename}")
    
    return oof_preds, test_preds, last_model


# # --- Model 1: XGBoost ---
# print("Training XGBoost...")
# xgb_params = {
#     'n_estimators': 3000,
#     'learning_rate': 0.015,
#     'max_depth': 8,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'objective': 'binary:logistic',
#     'eval_metric': 'auc',
#     'n_jobs': -1,
#     'random_state': SEED,
#     # 'tree_method': 'gpu_hist', # Uncomment if using GPU
#     # 'predictor': 'gpu_predictor'   # GPU ENABLED
# }

# oof_xgb, pred_xgb, model_xgb = train_and_submit(
#     'XGBoost', xgb_params, X_encoded, y, X_test_encoded, 'submission_xgb.csv'
# )


# # Feature Importance Plot (XGBoost)
# importances = pd.DataFrame({
#     'Feature': X_encoded.columns,
#     'Importance': model_xgb.feature_importances_
# }).sort_values(by='Importance', ascending=False)

# plt.figure(figsize=(10, 8))
# sns.barplot(x='Importance', y='Feature', data=importances.head(20), palette='viridis')
# plt.title('Top 20 Feature Importance (XGBoost)')
# plt.show()


# # --- Model 2: LightGBM ---
# print("\nTraining LightGBM...")
# lgb_params = {
#     'n_estimators': 3000,
#     'learning_rate': 0.015,
#     'num_leaves': 64,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'objective': 'binary',
#     'metric': 'auc',
#     'n_jobs': -1,
#     'random_state': SEED,
#     'verbosity': -1,
#     'device': 'cpu'                # CPU is safer for standard Kaggle env
# }

# oof_lgb, pred_lgb, _ = train_and_submit(
#     'LightGBM', lgb_params, X_encoded, y, X_test_encoded, 'submission_lgb.csv'
# )


# --- Model 3: CatBoost ---
cat_params = {
    'iterations': 3000,
    'learning_rate': 0.015,
    'depth': 8,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': SEED,
    'verbose': False,
    'allow_writing_files': False,
    'task_type': 'GPU',            # GPU ENABLED
    'devices': '0'                 # Use first GPU}
}
oof_cat, pred_cat, _ = train_and_submit(
    'CatBoost', cat_params, X_encoded, y, X_test_encoded, 'submission.csv'
)


# # ENSEMBLE OPTIMIZATION
# print("\n--- Optimizing Ensemble Weights ---")

# def minimize_auc(weights):
#     # Normalize weights so they sum to 1
#     w = weights / np.sum(weights)
#     # Create blended prediction
#     blend = (w[0] * oof_xgb) + (w[1] * oof_lgb) + (w[2] * oof_cat)
#     return -roc_auc_score(y, blend)

# # Initial guess: Equal weights
# init_weights = [0.33, 0.33, 0.33]
# bounds = [(0, 1), (0, 1), (0, 1)]

# res = minimize(minimize_auc, init_weights, bounds=bounds, method='SLSQP', tol=1e-6)
# final_weights = res.x / np.sum(res.x)

# print(f"Optimal Weights Found:")
# print(f"  XGBoost:  {final_weights[0]:.4f}")
# print(f"  LightGBM: {final_weights[1]:.4f}")
# print(f"  CatBoost: {final_weights[2]:.4f}")
# print(f"Best Ensemble AUC: {-res.fun:.5f}")


# # SUBMISSION
# final_test_preds = (final_weights[0] * pred_xgb) + \
#                    (final_weights[1] * pred_lgb) + \
#                    (final_weights[2] * pred_cat)

# submission = pd.DataFrame({
#     'id': submission_id,
#     TARGET: final_test_preds
# })

# submission.to_csv('submission.csv', index=False)
# print("\nsubmission.csv saved successfully.")
# print(submission.head())

