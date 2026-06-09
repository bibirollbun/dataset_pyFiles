import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')

print("âœ… Libraries loaded successfully")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"LightGBM version: {lgb.__version__}")


# Load data
print("ğŸ“‚ Loading datasets...\n")

train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")
print(f"âœ… Sample submission shape: {sample_sub.shape}")

print("\nğŸ“Š Train columns:")
print(train.columns.tolist())

print("\nğŸ“Š First 3 rows of train:")
display(train.head(3))

print("\nğŸ“Š Sample submission format:")
display(sample_sub.head(3))


print("ğŸ”� TRAIN DATA INFO\n" + "="*50)
print(train.info())

print("\nğŸ”� TRAIN DATA TYPES\n" + "="*50)
print(train.dtypes.value_counts())

print("\nğŸ”� TARGET VARIABLE\n" + "="*50)
if 'diagnosed_diabetes' in train.columns:
    target_col = 'diagnosed_diabetes'
    print(f"Target: {target_col}")
    print(f"Unique values: {train[target_col].unique()}")
    print(f"Value counts:\n{train[target_col].value_counts()}")
    print(f"Class balance: {train[target_col].value_counts(normalize=True)}")
else:
    # Find target column
    possible_targets = [col for col in train.columns if col not in test.columns and col != 'id']
    print(f"Possible target columns: {possible_targets}")
    if len(possible_targets) > 0:
        target_col = possible_targets[0]
        print(f"\nUsing '{target_col}' as target")


print("ğŸ”� MISSING VALUES\n" + "="*50)
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()

print("Train missing values:")
if missing_train.sum() > 0:
    print(missing_train[missing_train > 0].sort_values(ascending=False))
    print(f"\nTotal missing: {missing_train.sum()} ({missing_train.sum() / (train.shape[0] * train.shape[1]) * 100:.2f}%)")
else:
    print("âœ… No missing values in train")

print("\nTest missing values:")
if missing_test.sum() > 0:
    print(missing_test[missing_test > 0].sort_values(ascending=False))
    print(f"\nTotal missing: {missing_test.sum()} ({missing_test.sum() / (test.shape[0] * test.shape[1]) * 100:.2f}%)")
else:
    print("âœ… No missing values in test")


print("ğŸ”� BASIC STATISTICS\n" + "="*50)
print("\nTrain statistics:")
display(train.describe())

print("\nTest statistics:")
display(test.describe())


# Identify target column
if 'diagnosed_diabetes' in train.columns:
    target_col = 'diagnosed_diabetes'
else:
    target_col = [col for col in train.columns if col not in test.columns and col != 'id'][0]

print(f"ğŸ“Š Target variable: {target_col}\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
train[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
axes[0].set_title(f'{target_col} Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel(target_col)
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(train[target_col].value_counts()):
    axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

# Pie chart
train[target_col].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                       colors=['#3498db', '#e74c3c'], startangle=90)
axes[1].set_title(f'{target_col} Proportion', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

print(f"\nâœ… Class balance ratio: {train[target_col].value_counts(normalize=True).values}")


# Identify feature columns
feature_cols = [col for col in train.columns if col not in ['id', target_col]]

print(f"ğŸ“Š Feature columns ({len(feature_cols)} total):\n")
print(feature_cols)

# Separate numeric and categorical
numeric_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nğŸ”¢ Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"ğŸ�·ï¸�  Categorical features ({len(categorical_cols)}): {categorical_cols}")


# Numeric features distribution
if len(numeric_cols) > 0:
    print("ğŸ“Š Numeric Features Distribution\n" + "="*50)
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(train[col].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col}\n(mean={train[col].mean():.2f}, std={train[col].std():.2f})', 
                           fontsize=10, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(axis='y', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


# Categorical features
if len(categorical_cols) > 0:
    print("ğŸ“Š Categorical Features\n" + "="*50)
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {train[col].nunique()}")
        print(f"  Top 10 values:\n{train[col].value_counts().head(10)}")


if len(numeric_cols) > 0:
    print("ğŸ“Š Correlation with Target\n" + "="*50)
    
    correlations = train[numeric_cols + [target_col]].corr()[target_col].drop(target_col).sort_values(ascending=False)
    
    print("\nTop 10 positive correlations:")
    print(correlations.head(10))
    
    print("\nTop 10 negative correlations:")
    print(correlations.tail(10))
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = train[numeric_cols + [target_col]].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


print("ğŸ”§ Feature Engineering\n" + "="*50)

# Make copies
train_fe = train.copy()
test_fe = test.copy()

print(f"Starting shape - Train: {train_fe.shape}, Test: {test_fe.shape}")

# Handle categorical features if any
if len(categorical_cols) > 0:
    print(f"\nEncoding {len(categorical_cols)} categorical features...")
    for col in categorical_cols:
        # Label encoding
        combined = pd.concat([train_fe[col], test_fe[col]], axis=0)
        categories = combined.unique()
        cat_map = {cat: idx for idx, cat in enumerate(categories)}
        
        train_fe[col] = train_fe[col].map(cat_map)
        test_fe[col] = test_fe[col].map(cat_map)
        
        print(f"  âœ… {col}: {len(categories)} unique values")

print(f"\nâœ… Final shape - Train: {train_fe.shape}, Test: {test_fe.shape}")
print(f"âœ… All features are now numeric")


print("ğŸš€ LightGBM Training\n" + "="*50)

# Prepare data
X = train_fe[feature_cols].values
y = train_fe[target_col].values
X_test = test_fe[feature_cols].values

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"\nTarget distribution: {np.bincount(y.astype(int))}")

# LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

print(f"\nğŸ“‹ LightGBM Parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")


# Cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_scores = []
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print(f"\nğŸ”„ {n_splits}-Fold Cross-Validation\n" + "="*50)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nğŸ“� Fold {fold}/{n_splits}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Train target dist: {np.bincount(y_train.astype(int))}")
    print(f"  Val target dist: {np.bincount(y_val.astype(int))}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    # Predict
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Calculate score
    fold_score = roc_auc_score(y_val, val_pred)
    cv_scores.append(fold_score)
    
    # Store predictions
    oof_preds[val_idx] = val_pred
    test_preds += test_pred / n_splits
    
    print(f"  âœ… Fold {fold} AUC: {fold_score:.5f}")
    print(f"  Best iteration: {model.best_iteration}")

print(f"\n{'='*50}")
print(f"ğŸ“Š CV Results:")
print(f"  Mean AUC: {np.mean(cv_scores):.5f}")
print(f"  Std AUC: {np.std(cv_scores):.5f}")
print(f"  Fold scores: {[f'{s:.5f}' for s in cv_scores]}")
print(f"\n  OOF AUC: {roc_auc_score(y, oof_preds):.5f}")


# Get feature importance from last model
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("ğŸ“Š Top 20 Important Features:\n")
print(importance_df.head(20))

# Plot
plt.figure(figsize=(10, 8))
top_n = min(20, len(importance_df))
sns.barplot(data=importance_df.head(top_n), y='feature', x='importance', palette='viridis')
plt.title('Top 20 Feature Importance (Gain)', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


print("ğŸ“� Creating Submission\n" + "="*50)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_preds
})

print(f"Submission shape: {submission.shape}")
print(f"\nFirst 5 rows:")
display(submission.head())

print(f"\nPrediction statistics:")
print(submission['diagnosed_diabetes'].describe())

# Save
submission.to_csv('submission.csv', index=False)
print(f"\nâœ… Submission saved to submission.csv")

# Verify format
print(f"\nğŸ”� Format verification:")
print(f"  Columns: {submission.columns.tolist()}")
print(f"  Shape matches sample_submission: {submission.shape == sample_sub.shape}")
print(f"  IDs match: {(submission['id'] == sample_sub['id']).all()}")

