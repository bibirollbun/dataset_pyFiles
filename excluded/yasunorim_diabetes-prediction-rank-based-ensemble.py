import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("âœ… Libraries loaded")
print(f"LightGBM: {lgb.__version__}")


# Load data
print("ğŸ“‚ Loading datasets...\n")

train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(f"âœ… Train: {train.shape}, Test: {test.shape}")

# Identify target and features
target_col = 'diagnosed_diabetes'
feature_cols = [col for col in train.columns if col not in ['id', target_col]]
categorical_cols = train[feature_cols].select_dtypes(include=['object']).columns.tolist()

print(f"\nğŸ“Š Features: {len(feature_cols)} ({len(categorical_cols)} categorical)")
print(f"Target balance: {train[target_col].value_counts(normalize=True).values}")


print("ğŸ”§ Encoding categorical features...\n")

train_fe = train.copy()
test_fe = test.copy()

for col in categorical_cols:
    combined = pd.concat([train_fe[col], test_fe[col]], axis=0)
    categories = combined.unique()
    cat_map = {cat: idx for idx, cat in enumerate(categories)}
    
    train_fe[col] = train_fe[col].map(cat_map)
    test_fe[col] = test_fe[col].map(cat_map)
    
    print(f"  âœ… {col}: {len(categories)} categories")

print(f"\nâœ… All features ready for training")


print("ğŸš€ Model 1 Training (Main Model)\n" + "="*50)

# Prepare data
X = train_fe[feature_cols].values
y = train_fe[target_col].values
X_test = test_fe[feature_cols].values

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")

# Model 1 parameters (seed=42)
params_model1 = {
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

print(f"\nğŸ“‹ Model 1 Parameters (seed=42):")
for k, v in params_model1.items():
    print(f"  {k}: {v}")


# Cross-validation for Model 1
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

model1_cv_scores = []
model1_oof_preds = np.zeros(len(X))
model1_test_preds = np.zeros(len(X_test))

print(f"\nğŸ”„ Model 1: {n_splits}-Fold Cross-Validation\n" + "="*50)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nğŸ“� Fold {fold}/{n_splits}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        params_model1,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    # Predict
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Calculate score
    fold_score = roc_auc_score(y_val, val_pred)
    model1_cv_scores.append(fold_score)
    
    # Store predictions
    model1_oof_preds[val_idx] = val_pred
    model1_test_preds += test_pred / n_splits
    
    print(f"  âœ… Fold {fold} AUC: {fold_score:.5f} (iter: {model.best_iteration})")

print(f"\n{'='*50}")
print(f"ğŸ“Š Model 1 CV Results:")
print(f"  Mean AUC: {np.mean(model1_cv_scores):.5f} Â± {np.std(model1_cv_scores):.5f}")
print(f"  OOF AUC: {roc_auc_score(y, model1_oof_preds):.5f}")
print(f"  Fold scores: {[f'{s:.5f}' for s in model1_cv_scores]}")


print("ğŸš€ Model 2 Training (Diversity Model)\n" + "="*50)

# Model 2 parameters (seed=123 for diversity)
params_model2 = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 123  # Different seed for diversity
}

print(f"ğŸ“‹ Model 2 Parameters (seed=123):")
for k, v in params_model2.items():
    print(f"  {k}: {v}")


# Cross-validation for Model 2
model2_cv_scores = []
model2_oof_preds = np.zeros(len(X))
model2_test_preds = np.zeros(len(X_test))

print(f"\nğŸ”„ Model 2: {n_splits}-Fold Cross-Validation\n" + "="*50)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nğŸ“� Fold {fold}/{n_splits}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        params_model2,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    # Predict
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Calculate score
    fold_score = roc_auc_score(y_val, val_pred)
    model2_cv_scores.append(fold_score)
    
    # Store predictions
    model2_oof_preds[val_idx] = val_pred
    model2_test_preds += test_pred / n_splits
    
    print(f"  âœ… Fold {fold} AUC: {fold_score:.5f} (iter: {model.best_iteration})")

print(f"\n{'='*50}")
print(f"ğŸ“Š Model 2 CV Results:")
print(f"  Mean AUC: {np.mean(model2_cv_scores):.5f} Â± {np.std(model2_cv_scores):.5f}")
print(f"  OOF AUC: {roc_auc_score(y, model2_oof_preds):.5f}")
print(f"  Fold scores: {[f'{s:.5f}' for s in model2_cv_scores]}")


print("ğŸ�¯ Rank-Based Blending\n" + "="*50)

# Create dataframes for blending
df_model1 = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': model1_test_preds
})

df_model2 = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': model2_test_preds
})

# Apply rank transformation with weights
weight_model1 = 1.0
weight_model2 = 0.5

df_model1['weighted_pred'] = df_model1['diagnosed_diabetes'].rank(pct=True) * weight_model1
df_model2['weighted_pred'] = df_model2['diagnosed_diabetes'].rank(pct=True) * weight_model2

print(f"Model 1 predictions:")
print(f"  Raw range: [{model1_test_preds.min():.4f}, {model1_test_preds.max():.4f}]")
print(f"  Rank range: [{df_model1['weighted_pred'].min():.4f}, {df_model1['weighted_pred'].max():.4f}]")

print(f"\nModel 2 predictions:")
print(f"  Raw range: [{model2_test_preds.min():.4f}, {model2_test_preds.max():.4f}]")
print(f"  Rank range: [{df_model2['weighted_pred'].min():.4f}, {df_model2['weighted_pred'].max():.4f}]")

# Merge and average
df_blend = df_model1[['id', 'weighted_pred']].merge(
    df_model2[['id', 'weighted_pred']],
    on='id',
    how='left',
    suffixes=('_1', '_2')
)

# Calculate weighted average
df_blend['diagnosed_diabetes'] = (
    df_blend['weighted_pred_1'] + df_blend['weighted_pred_2']
) / (weight_model1 + weight_model2)

print(f"\nBlended predictions:")
print(f"  Range: [{df_blend['diagnosed_diabetes'].min():.4f}, {df_blend['diagnosed_diabetes'].max():.4f}]")
print(f"  Mean: {df_blend['diagnosed_diabetes'].mean():.4f}")
print(f"  Std: {df_blend['diagnosed_diabetes'].std():.4f}")

# Verify OOF blending
df_oof_model1 = pd.DataFrame({'diagnosed_diabetes': model1_oof_preds})
df_oof_model2 = pd.DataFrame({'diagnosed_diabetes': model2_oof_preds})

df_oof_model1['weighted_pred'] = df_oof_model1['diagnosed_diabetes'].rank(pct=True) * weight_model1
df_oof_model2['weighted_pred'] = df_oof_model2['diagnosed_diabetes'].rank(pct=True) * weight_model2

oof_blend = (df_oof_model1['weighted_pred'] + df_oof_model2['weighted_pred']) / (weight_model1 + weight_model2)
oof_blend_auc = roc_auc_score(y, oof_blend)

print(f"\nğŸ“Š OOF Blending Results:")
print(f"  Model 1 OOF AUC: {roc_auc_score(y, model1_oof_preds):.5f}")
print(f"  Model 2 OOF AUC: {roc_auc_score(y, model2_oof_preds):.5f}")
print(f"  Blended OOF AUC: {oof_blend_auc:.5f}")
print(f"\nâœ… Rank-based blending completed")


# Feature importance from last model
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
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


print("ğŸ“� Creating Submission\n" + "="*50)

# Use blended predictions
submission = df_blend[['id', 'diagnosed_diabetes']].copy()

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
print(f"  Shape matches: {submission.shape == sample_sub.shape}")
print(f"  IDs match: {(submission['id'] == sample_sub['id']).all()}")

