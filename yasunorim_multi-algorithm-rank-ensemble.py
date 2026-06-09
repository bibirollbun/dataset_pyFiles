import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("âœ… Libraries loaded")
print(f"LightGBM: {lgb.__version__}")
print(f"XGBoost: {xgb.__version__}")


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

# Prepare data
X = train_fe[feature_cols].values
y = train_fe[target_col].values
X_test = test_fe[feature_cols].values

print(f"\nâœ… Data prepared")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")


print("ğŸš€ Model 1: LightGBM\n" + "="*50)

# LightGBM parameters
lgb_params = {
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

# Cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_cv_scores = []
lgb_oof_preds = np.zeros(len(X))
lgb_test_preds = np.zeros(len(X_test))

print(f"ğŸ”„ {n_splits}-Fold Cross-Validation\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{n_splits}", end=" ")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=['valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    fold_score = roc_auc_score(y_val, val_pred)
    lgb_cv_scores.append(fold_score)
    lgb_oof_preds[val_idx] = val_pred
    lgb_test_preds += test_pred / n_splits
    
    print(f"AUC: {fold_score:.5f}")

print(f"\nğŸ“Š LightGBM Results:")
print(f"  Mean AUC: {np.mean(lgb_cv_scores):.5f} Â± {np.std(lgb_cv_scores):.5f}")
print(f"  OOF AUC: {roc_auc_score(y, lgb_oof_preds):.5f}")


print("\nğŸš€ Model 2: XGBoost\n" + "="*50)

# XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0
}

# Cross-validation
xgb_cv_scores = []
xgb_oof_preds = np.zeros(len(X))
xgb_test_preds = np.zeros(len(X_test))

print(f"ğŸ”„ {n_splits}-Fold Cross-Validation\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{n_splits}", end=" ")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=0
    )
    
    val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_pred = model.predict(xgb.DMatrix(X_test), iteration_range=(0, model.best_iteration + 1))
    
    fold_score = roc_auc_score(y_val, val_pred)
    xgb_cv_scores.append(fold_score)
    xgb_oof_preds[val_idx] = val_pred
    xgb_test_preds += test_pred / n_splits
    
    print(f"AUC: {fold_score:.5f}")

print(f"\nğŸ“Š XGBoost Results:")
print(f"  Mean AUC: {np.mean(xgb_cv_scores):.5f} Â± {np.std(xgb_cv_scores):.5f}")
print(f"  OOF AUC: {roc_auc_score(y, xgb_oof_preds):.5f}")


print("\nğŸš€ Model 3: CatBoost\n" + "="*50)

# CatBoost parameters
cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 0,
    'early_stopping_rounds': 50
}

# Cross-validation
cat_cv_scores = []
cat_oof_preds = np.zeros(len(X))
cat_test_preds = np.zeros(len(X_test))

print(f"ğŸ”„ {n_splits}-Fold Cross-Validation\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{n_splits}", end=" ")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    
    model = CatBoostClassifier(**cat_params)
    model.fit(train_pool, eval_set=val_pool, verbose=0)
    
    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    
    fold_score = roc_auc_score(y_val, val_pred)
    cat_cv_scores.append(fold_score)
    cat_oof_preds[val_idx] = val_pred
    cat_test_preds += test_pred / n_splits
    
    print(f"AUC: {fold_score:.5f}")

print(f"\nğŸ“Š CatBoost Results:")
print(f"  Mean AUC: {np.mean(cat_cv_scores):.5f} Â± {np.std(cat_cv_scores):.5f}")
print(f"  OOF AUC: {roc_auc_score(y, cat_oof_preds):.5f}")


print("\nğŸ�¯ Rank-Based Blending\n" + "="*50)

# Create dataframes for each model's predictions
df_lgb = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': lgb_test_preds
})

df_xgb = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': xgb_test_preds
})

df_cat = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': cat_test_preds
})

# Apply rank transformation with weights
weight_lgb = 1.0
weight_xgb = 0.8
weight_cat = 0.6

df_lgb['weighted_pred'] = df_lgb['diagnosed_diabetes'].rank(pct=True) * weight_lgb
df_xgb['weighted_pred'] = df_xgb['diagnosed_diabetes'].rank(pct=True) * weight_xgb
df_cat['weighted_pred'] = df_cat['diagnosed_diabetes'].rank(pct=True) * weight_cat

print(f"Rank transformation applied:")
print(f"  LightGBM weight: {weight_lgb}")
print(f"  XGBoost weight: {weight_xgb}")
print(f"  CatBoost weight: {weight_cat}")

# Merge all predictions
df_blend = df_lgb[['id', 'weighted_pred']].copy()
df_blend = df_blend.merge(
    df_xgb[['id', 'weighted_pred']], 
    on='id', 
    suffixes=('_lgb', '_xgb')
)
df_blend = df_blend.merge(
    df_cat[['id', 'weighted_pred']], 
    on='id'
)
df_blend.rename(columns={'weighted_pred': 'weighted_pred_cat'}, inplace=True)

# Calculate weighted average
df_blend['diagnosed_diabetes'] = (
    df_blend['weighted_pred_lgb'] + 
    df_blend['weighted_pred_xgb'] + 
    df_blend['weighted_pred_cat']
) / (weight_lgb + weight_xgb + weight_cat)

print(f"\nBlended predictions:")
print(f"  Range: [{df_blend['diagnosed_diabetes'].min():.4f}, {df_blend['diagnosed_diabetes'].max():.4f}]")
print(f"  Mean: {df_blend['diagnosed_diabetes'].mean():.4f}")
print(f"  Std: {df_blend['diagnosed_diabetes'].std():.4f}")

# Calculate OOF blending performance
df_oof_lgb = pd.DataFrame({'pred': lgb_oof_preds})
df_oof_xgb = pd.DataFrame({'pred': xgb_oof_preds})
df_oof_cat = pd.DataFrame({'pred': cat_oof_preds})

oof_lgb_rank = df_oof_lgb['pred'].rank(pct=True) * weight_lgb
oof_xgb_rank = df_oof_xgb['pred'].rank(pct=True) * weight_xgb
oof_cat_rank = df_oof_cat['pred'].rank(pct=True) * weight_cat

oof_blend = (oof_lgb_rank + oof_xgb_rank + oof_cat_rank) / (weight_lgb + weight_xgb + weight_cat)
oof_blend_auc = roc_auc_score(y, oof_blend)

print(f"\nğŸ“Š OOF Blending Results:")
print(f"  LightGBM OOF AUC: {roc_auc_score(y, lgb_oof_preds):.5f}")
print(f"  XGBoost OOF AUC: {roc_auc_score(y, xgb_oof_preds):.5f}")
print(f"  CatBoost OOF AUC: {roc_auc_score(y, cat_oof_preds):.5f}")
print(f"  Blended OOF AUC: {oof_blend_auc:.5f}")
print(f"\nâœ… Rank-based blending completed")


# Visualize model performance comparison
model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'Blended']
oof_aucs = [
    roc_auc_score(y, lgb_oof_preds),
    roc_auc_score(y, xgb_oof_preds),
    roc_auc_score(y, cat_oof_preds),
    oof_blend_auc
]

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, oof_aucs, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
plt.ylabel('OOF AUC', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([0.72, 0.73])
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, auc in zip(bars, oof_aucs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{auc:.5f}',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


print("\nğŸ“� Creating Submission\n" + "="*50)

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

