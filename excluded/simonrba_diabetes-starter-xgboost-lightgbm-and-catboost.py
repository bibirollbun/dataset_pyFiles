# ============================================================================
# LIBRARY IMPORTS
# ============================================================================

# Core Data Science Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Gradient Boosting Models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

print("All libraries imported successfully!")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Cross-validation configuration
N_FOLDS = 5

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
%matplotlib inline

print(f"Configuration:")
print(f"   - Random Seed: {RANDOM_SEED}")
print(f"   - CV Folds: {N_FOLDS}")


# ============================================================================
# LOAD DATA
# ============================================================================

# Detect environment (Kaggle or local)
if os.path.exists('/kaggle/input'):
    DATA_PATH = '/kaggle/input/playground-series-s5e12/'
    print("Running on Kaggle")
else:
    DATA_PATH = './'
    print("Running locally")

# Load datasets
train = pd.read_csv(DATA_PATH + 'train.csv')
test = pd.read_csv(DATA_PATH + 'test.csv')
sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')

print(f"\nDataset Shapes:")
print(f"   - Training:   {train.shape[0]:,} rows × {train.shape[1]} columns")
print(f"   - Test:       {test.shape[0]:,} rows × {test.shape[1]} columns")
print(f"   - Submission: {sample_submission.shape[0]:,} rows × {sample_submission.shape[1]} columns")


# Display first few rows
print("\nFirst 5 rows of training data:")
train.head()


# Display column information
print("\nColumn Information:")
train.info()


# ============================================================================
# TARGET DISTRIBUTION
# ============================================================================

print("Target Variable: diagnosed_diabetes")
print("\nClass Distribution:")
print(train['diagnosed_diabetes'].value_counts())
print("\nClass Proportions:")
print(train['diagnosed_diabetes'].value_counts(normalize=True))

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train['diagnosed_diabetes'].value_counts().plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
axes[0].set_title('Target Distribution (Count)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Diagnosed Diabetes', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)

train['diagnosed_diabetes'].value_counts(normalize=True).plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'])
axes[1].set_title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Diagnosed Diabetes', fontsize=12)
axes[1].set_ylabel('Proportion', fontsize=12)
axes[1].set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)

plt.tight_layout()
plt.show()


# ============================================================================
# DATA QUALITY ASSESSMENT
# ============================================================================

print("Checking for Missing Values...\n")

missing_train = train.isnull().sum()
missing_test = test.isnull().sum()

if missing_train.sum() == 0:
    print("No missing values in training set!")
else:
    print("Missing values found in training set:")
    print(missing_train[missing_train > 0])

if missing_test.sum() == 0:
    print("No missing values in test set!")
else:
    print("\nMissing values found in test set:")
    print(missing_test[missing_test > 0])

print(f"\nDuplicate rows in training set: {train.duplicated().sum()}")
print(f"Duplicate rows in test set: {test.duplicated().sum()}")


# ============================================================================
# IDENTIFY FEATURE TYPES
# ============================================================================

id_col = 'id'
target_col = 'diagnosed_diabetes'
feature_cols = [col for col in train.columns if col not in [id_col, target_col]]

numeric_features = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

print("Feature Summary:")
print(f"   - Total features: {len(feature_cols)}")
print(f"   - Numeric features: {len(numeric_features)}")
print(f"   - Categorical features: {len(categorical_features)}")
print(f"\nNumeric Features: {numeric_features}")
print(f"\nCategorical Features: {categorical_features}")


# ============================================================================
# CORRELATION MATRIX
# ============================================================================

if len(numeric_features) > 0:
    correlation_matrix = train[numeric_features + [target_col]].corr()
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Features Correlated with Target:\n")
    target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
    top_corr = target_corr[target_corr.index != target_col].head(10)
    
    for idx, (feature, corr) in enumerate(top_corr.items(), 1):
        print(f"   {idx:2d}. {feature:30s} -> {corr:.4f}")


# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================

def create_features(df, is_train=True):
    """
    Comprehensive feature engineering pipeline.
    """
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    if 'diagnosed_diabetes' in numeric_cols:
        numeric_cols.remove('diagnosed_diabetes')
    
    print(f"\n{'='*70}")
    print(f"FEATURE ENGINEERING {'(TRAINING)' if is_train else '(TEST)'}")
    print(f"{'='*70}")
    print(f"\nStarting with {len(numeric_cols)} numeric features")
    
    # Polynomial features
    print("\nCreating polynomial features...")
    poly_count = 0
    for col in numeric_cols[:10]:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_cubed'] = df[col] ** 3
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        df[f'{col}_log1p'] = np.log1p(np.abs(df[col]))
        poly_count += 4
    print(f"   Created {poly_count} polynomial features")
    
    # Interaction features
    print("\nCreating interaction features...")
    interaction_count = 0
    for i in range(min(5, len(numeric_cols))):
        for j in range(i+1, min(5, len(numeric_cols))):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)
            interaction_count += 2
    print(f"   Created {interaction_count} interaction features")
    
    # Statistical aggregations
    print("\nCreating statistical features...")
    df['feature_sum'] = df[numeric_cols].sum(axis=1)
    df['feature_mean'] = df[numeric_cols].mean(axis=1)
    df['feature_std'] = df[numeric_cols].std(axis=1)
    df['feature_min'] = df[numeric_cols].min(axis=1)
    df['feature_max'] = df[numeric_cols].max(axis=1)
    df['feature_range'] = df['feature_max'] - df['feature_min']
    print(f"   Created 6 statistical features")
    
    print(f"\n{'='*70}")
    print(f"Feature engineering complete!")
    print(f"   - New shape: {df.shape}")
    print(f"{'='*70}")
    
    return df

# Apply feature engineering
train_fe = create_features(train, is_train=True)
test_fe = create_features(test, is_train=False)


# ============================================================================
# CLUSTERING FEATURES
# ============================================================================

print("\nCreating clustering features...\n")

X_cluster = train_fe.select_dtypes(include=[np.number]).drop(
    columns=['id', 'diagnosed_diabetes'], errors='ignore'
)
X_cluster_test = test_fe.select_dtypes(include=[np.number]).drop(
    columns=['id'], errors='ignore'
)

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster.fillna(0))
X_cluster_test_scaled = scaler_cluster.transform(X_cluster_test.fillna(0))

cluster_configs = [3, 5, 8]
for n_clusters in cluster_configs:
    print(f"   K-Means with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    train_fe[f'cluster_{n_clusters}'] = kmeans.fit_predict(X_cluster_scaled)
    test_fe[f'cluster_{n_clusters}'] = kmeans.predict(X_cluster_test_scaled)
    train_fe[f'cluster_{n_clusters}_dist'] = kmeans.transform(X_cluster_scaled).min(axis=1)
    test_fe[f'cluster_{n_clusters}_dist'] = kmeans.transform(X_cluster_test_scaled).min(axis=1)

print("\nClustering features created!")


# ============================================================================
# PCA FEATURES
# ============================================================================

print("\nCreating PCA features...\n")

N_COMPONENTS = 10
pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
pca_train = pca.fit_transform(X_cluster_scaled)
pca_test = pca.transform(X_cluster_test_scaled)

for i in range(N_COMPONENTS):
    train_fe[f'pca_{i}'] = pca_train[:, i]
    test_fe[f'pca_{i}'] = pca_test[:, i]

total_variance = pca.explained_variance_ratio_.sum()
print(f"Created {N_COMPONENTS} PCA components")
print(f"Total variance explained: {total_variance:.2%}")


# ============================================================================
# PREPARE DATA FOR MODELING
# ============================================================================

print("\nPreparing data for modeling...\n")

X = train_fe.drop(columns=['id', 'diagnosed_diabetes'])
y = train_fe['diagnosed_diabetes']
X_test = test_fe.drop(columns=['id'])

print(f"Initial shapes:")
print(f"   - X: {X.shape}")
print(f"   - y: {y.shape}")
print(f"   - X_test: {X_test.shape}")

# Handle categorical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if len(categorical_cols) > 0:
    print(f"\nEncoding {len(categorical_cols)} categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

X = X.fillna(0)
X_test = X_test.fillna(0)

print(f"\nData preparation complete!")
print(f"Final shape: {X.shape}")


# ============================================================================
# XGBOOST MODEL
# ============================================================================

print("\nTraining XGBoost model...\n")

# Good default parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': RANDOM_SEED,
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X, y)

xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=N_FOLDS, scoring='roc_auc', n_jobs=-1)

print(f"XGBoost Performance:")
print(f"   - CV AUC: {xgb_cv_scores.mean():.6f} (+/- {xgb_cv_scores.std():.6f})")
print(f"   - Min: {xgb_cv_scores.min():.6f}, Max: {xgb_cv_scores.max():.6f}")


# ============================================================================
# LIGHTGBM MODEL
# ============================================================================

print("\nTraining LightGBM model...\n")

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'random_state': RANDOM_SEED,
    'verbose': -1,
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X, y)

lgb_cv_scores = cross_val_score(lgb_model, X, y, cv=N_FOLDS, scoring='roc_auc', n_jobs=-1)

print(f"LightGBM Performance:")
print(f"   - CV AUC: {lgb_cv_scores.mean():.6f} (+/- {lgb_cv_scores.std():.6f})")
print(f"   - Min: {lgb_cv_scores.min():.6f}, Max: {lgb_cv_scores.max():.6f}")


# ============================================================================
# CATBOOST MODEL
# ============================================================================

print("\nTraining CatBoost model...\n")

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': RANDOM_SEED,
    'verbose': False,
    'iterations': 300,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
}

cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(X, y)

cat_cv_scores = cross_val_score(cat_model, X, y, cv=N_FOLDS, scoring='roc_auc', n_jobs=-1)

print(f"CatBoost Performance:")
print(f"   - CV AUC: {cat_cv_scores.mean():.6f} (+/- {cat_cv_scores.std():.6f})")
print(f"   - Min: {cat_cv_scores.min():.6f}, Max: {cat_cv_scores.max():.6f}")


# ============================================================================
# STACKING ENSEMBLE
# ============================================================================

print("\nBuilding stacking ensemble...\n")

estimators = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('cat', cat_model)
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
    cv=N_FOLDS,
    n_jobs=-1
)

stacking_model.fit(X, y)

stacking_cv_scores = cross_val_score(stacking_model, X, y, cv=N_FOLDS, scoring='roc_auc', n_jobs=-1)

print(f"Stacking Ensemble Performance:")
print(f"   - CV AUC: {stacking_cv_scores.mean():.6f} (+/- {stacking_cv_scores.std():.6f})")
print(f"   - Min: {stacking_cv_scores.min():.6f}, Max: {stacking_cv_scores.max():.6f}")


# ============================================================================
# MODEL COMPARISON
# ============================================================================

model_comparison = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Stacking'],
    'CV AUC': [
        xgb_cv_scores.mean(),
        lgb_cv_scores.mean(),
        cat_cv_scores.mean(),
        stacking_cv_scores.mean()
    ],
    'Std': [
        xgb_cv_scores.std(),
        lgb_cv_scores.std(),
        cat_cv_scores.std(),
        stacking_cv_scores.std()
    ]
}).sort_values('CV AUC', ascending=False)

print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70 + "\n")
print(model_comparison.to_string(index=False))

plt.figure(figsize=(10, 5))
plt.barh(model_comparison['Model'], model_comparison['CV AUC'], 
         color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
plt.xlabel('CV AUC Score', fontsize=12, fontweight='bold')
plt.title('Model Performance', fontsize=14, fontweight='bold')
plt.xlim([model_comparison['CV AUC'].min() - 0.01, 1.0])
for i, (model, score) in enumerate(zip(model_comparison['Model'], model_comparison['CV AUC'])):
    plt.text(score + 0.002, i, f'{score:.6f}', va='center')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nBest Model: {model_comparison.iloc[0]['Model']}")
print(f"CV AUC: {model_comparison.iloc[0]['CV AUC']:.6f}")


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_importance(model, name, top_n=20):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features - {name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    return importance_df

print("\nXGBoost Feature Importance:")
xgb_imp = plot_importance(xgb_model, 'XGBoost')
print(xgb_imp.head(10).to_string(index=False))


# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\nGenerating predictions...\n")

xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
cat_preds = cat_model.predict_proba(X_test)[:, 1]
stacking_preds = stacking_model.predict_proba(X_test)[:, 1]

print("Predictions generated!")

pred_stats = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Stacking'],
    'Mean': [xgb_preds.mean(), lgb_preds.mean(), cat_preds.mean(), stacking_preds.mean()],
    'Std': [xgb_preds.std(), lgb_preds.std(), cat_preds.std(), stacking_preds.std()],
    'Min': [xgb_preds.min(), lgb_preds.min(), cat_preds.min(), stacking_preds.min()],
    'Max': [xgb_preds.max(), lgb_preds.max(), cat_preds.max(), stacking_preds.max()]
})

print("\nPrediction Statistics:")
print(pred_stats.to_string(index=False))


# ============================================================================
# CREATE SUBMISSIONS
# ============================================================================

print("\nCreating submission files...\n")

# Stacking (primary)
pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': stacking_preds
}).to_csv('submission_stacking.csv', index=False)
print("Created: submission_stacking.csv (RECOMMENDED)")

# Individual models
pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': xgb_preds
}).to_csv('submission_xgb.csv', index=False)
print("Created: submission_xgb.csv")

pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': lgb_preds
}).to_csv('submission_lgb.csv', index=False)
print("Created: submission_lgb.csv")

pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': cat_preds
}).to_csv('submission_cat.csv', index=False)
print("Created: submission_cat.csv")

print("\nAll submission files created!")
print("Recommended: submission_stacking.csv")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("COMPETITION SUMMARY")
print("="*70)

print(f"\nDataset:")
print(f"   - Training: {len(train):,} samples")
print(f"   - Test: {len(test):,} samples")
print(f"   - Features: {X.shape[1]}")

print(f"\nModel Performance (CV AUC):")
print(f"   - XGBoost: {xgb_cv_scores.mean():.6f}")
print(f"   - LightGBM: {lgb_cv_scores.mean():.6f}")
print(f"   - CatBoost: {cat_cv_scores.mean():.6f}")
print(f"   - Stacking: {stacking_cv_scores.mean():.6f}")

print(f"\nSubmissions:")
print(f"   1. submission_stacking.csv (RECOMMENDED)")
print(f"   2. submission_xgb.csv")
print(f"   3. submission_lgb.csv")
print(f"   4. submission_cat.csv")

print("\n" + "="*70)
print("Notebook complete! Ready to submit!")
print("="*70)

