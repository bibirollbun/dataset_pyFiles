import pandas as pd
import numpy as np


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


# Get common columns
common_cols = set(df_train.columns) & set(df_test.columns)

# Check if they have the same columns (excluding target column)
train_cols = set(df_train.columns) - {'diagnosed_diabetes'}
test_cols = set(df_test.columns)

if train_cols == test_cols:
    print("✓ Both datasets have the same columns (excluding target)")
else:
    print("✗ Datasets do NOT have the same columns")

print(f"\nComparing data types for {len(common_cols)} common columns:")
print("-" * 60)

for col in sorted(common_cols):
    train_dtype = df_train[col].dtype
    test_dtype = df_test[col].dtype
    
    if train_dtype == test_dtype:
        status = "✓"
    else:
        status = "✗"
    
    print(f"{status} {col:<35} | Train: {str(train_dtype):<10} | Test: {str(test_dtype)}")


null_counts_train = df_train.isnull().sum()
total_nulls_train = null_counts_train.sum()

for col in df_train.columns:
    null_count = null_counts_train[col]
    if null_count > 0:
        print(f"{col:<35} | {null_count:>5} nulls")

if total_nulls_train == 0:
    print("✓ No null values found in train dataset")
else:
    print(f"\nTotal null values in train: {total_nulls_train}")


# Numerical features summary (limited output)
numerical_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['id', 'diagnosed_diabetes']]

print(f"Numerical features ({len(numerical_cols)}):")
print(df_train[numerical_cols].describe().T[['mean', 'std', 'min', 'max']].round(2))


# Categorical features
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical features ({len(categorical_cols)}):")
for col in categorical_cols:
    unique_count = df_train[col].nunique()
    print(f"  {col}: {unique_count} unique values")


# Correlation with target (top 10)
correlations = df_train[numerical_cols + ['diagnosed_diabetes']].corr()['diagnosed_diabetes'].abs().sort_values(ascending=False)
print("Top correlations with target:")
print(correlations.head(11).round(3))  # 11 to exclude target itself


# Check for outliers (IQR method) - only show features with significant outliers
outlier_summary = {}
for col in numerical_cols:
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df_train[col] < lower_bound) | (df_train[col] > upper_bound)).sum()
    if outliers > 0:
        outlier_summary[col] = outliers

if outlier_summary:
    print("Features with outliers (>1.5*IQR):")
    for col, count in sorted(outlier_summary.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = (count / len(df_train)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
else:
    print("No significant outliers detected")


# Target by categorical features (top categories only)
for col in categorical_cols:
    target_by_cat = df_train.groupby(col)['diagnosed_diabetes'].mean().sort_values(ascending=False)
    print(target_by_cat.head(5).round(3))
    print("\n")


# Ordinal encoding for variables with natural order
education_order = {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3}
income_order = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}

df_train['education_level_enc'] = df_train['education_level'].map(education_order)
df_train['income_level_enc'] = df_train['income_level'].map(income_order)

df_test['education_level_enc'] = df_test['education_level'].map(education_order)
df_test['income_level_enc'] = df_test['income_level'].map(income_order)

# Verify mappings worked
print(f"Education missing: {df_train['education_level_enc'].isna().sum()} train, {df_test['education_level_enc'].isna().sum()} test")
print(f"Income missing: {df_train['income_level_enc'].isna().sum()} train, {df_test['income_level_enc'].isna().sum()} test")


# Label encoding for nominal categorical variables (for tree-based models)
from sklearn.preprocessing import LabelEncoder

nominal_cols = ['gender', 'ethnicity', 'smoking_status', 'employment_status']
label_encoders = {}

for col in nominal_cols:
    le = LabelEncoder()
    # Fit on combined data to ensure consistency
    le.fit(pd.concat([df_train[col], df_test[col]]))
    df_train[f'{col}_enc'] = le.transform(df_train[col])
    df_test[f'{col}_enc'] = le.transform(df_test[col])
    label_encoders[col] = le

print("Label encoding complete for:", nominal_cols)



# Define feature columns for modeling
encoded_cat_cols = [f'{col}_enc' for col in categorical_cols]
feature_cols = numerical_cols + encoded_cat_cols

print(f"Total features: {len(feature_cols)}")
print(f"  Numerical: {len(numerical_cols)}")
print(f"  Categorical (encoded): {len(encoded_cat_cols)}")



# One-Hot Encoding for linear models (Logistic Regression, SVM, etc.)
# Only for nominal variables (no natural order)
nominal_cols = ['gender', 'ethnicity', 'smoking_status', 'employment_status']

# Create one-hot encoded features (drop_first=True to avoid multicollinearity)
df_train_ohe = pd.get_dummies(df_train[nominal_cols], drop_first=True)
df_test_ohe = pd.get_dummies(df_test[nominal_cols], drop_first=True)

# Ensure both have same columns
missing_in_test = set(df_train_ohe.columns) - set(df_test_ohe.columns)
for col in missing_in_test:
    df_test_ohe[col] = 0
df_test_ohe = df_test_ohe[df_train_ohe.columns]

# Add to dataframes
df_train = pd.concat([df_train, df_train_ohe], axis=1)
df_test = pd.concat([df_test, df_test_ohe], axis=1)

print(f"One-Hot features added: {list(df_train_ohe.columns)}")



# Feature sets for different model types
ohe_cols = list(df_train_ohe.columns)
ordinal_cols = ['education_level_enc', 'income_level_enc']

# For tree-based models (XGBoost, LightGBM, RandomForest) - can use label encoding
features_tree = numerical_cols + encoded_cat_cols

# For linear models (Logistic Regression, SVM) - use ordinal + one-hot
features_linear = numerical_cols + ordinal_cols + ohe_cols

print(f"Features for tree models: {len(features_tree)}")
print(f"Features for linear models: {len(features_linear)}")



from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Prepare data
X = df_train[features_tree]
y = df_train['diagnosed_diabetes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape[0]:,} samples")
print(f"Val: {X_val.shape[0]:,} samples")



# Prepare scaled data for linear models
X_linear = df_train[features_linear]
X_train_lin, X_val_lin, _, _ = train_test_split(X_linear, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_lin)
X_val_scaled = scaler.transform(X_val_lin)

print("Data scaled for linear models")



%%time
# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_val_scaled)
lr_acc = accuracy_score(y_val, lr_pred)
print(f"Logistic Regression: {lr_acc:.4f}")



%%time
# 2. Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_acc = accuracy_score(y_val, rf_pred)
print(f"Random Forest: {rf_acc:.4f}")



%%time
# 3. CatBoost (GPU)
cat = CatBoostClassifier(
    iterations=100, 
    depth=6, 
    learning_rate=0.1, 
    random_seed=42,
    task_type='GPU',
    devices='0',
    verbose=0
)
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_val)
cat_acc = accuracy_score(y_val, cat_pred)
print(f"CatBoost (GPU): {cat_acc:.4f}")



%%time
# 4. XGBoost (GPU)
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    random_state=42,
    tree_method='hist',
    device='cuda',
    verbosity=0
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_val)
xgb_acc = accuracy_score(y_val, xgb_pred)
print(f"XGBoost (GPU): {xgb_acc:.4f}")



%%time
# 5. LightGBM (GPU)
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    random_state=42, 
    device='gpu',  # <-- GPU acceleration
    verbose=-1
)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_val)
lgbm_acc = accuracy_score(y_val, lgbm_pred)
print(f"LightGBM (GPU): {lgbm_acc:.4f}")



# Results comparison
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'CatBoost', 'XGBoost', 'LightGBM'],
    'Accuracy': [lr_acc, rf_acc, cat_acc, xgb_acc, lgbm_acc]
}).sort_values('Accuracy', ascending=False)

print("=" * 40)
print("MODEL COMPARISON (GPU)")
print("=" * 40)
print(results.to_string(index=False))
print("=" * 40)
best_model = results.iloc[0]['Model']
best_acc = results.iloc[0]['Accuracy']
print(f"Best: {best_model} ({best_acc:.4f})")



# Imports already done in cell 17


%%time
# Hyperparameter tuning for XGBoost - ANTI-OVERFITTING settings
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),  # Reducido
        'max_depth': trial.suggest_int('max_depth', 2, 6),  # Más bajo
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8), 
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True), 
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.1, 5.0, log=True),
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda',
        'verbosity': 0
    }
    
    model = XGBClassifier(**params)
    # Usar 5 folds para mejor estimación
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

print(f"Best XGBoost accuracy (CV-5): {study_xgb.best_value:.4f}")



# Best XGBoost parameters found
print("Best XGBoost params:")
for k, v in study_xgb.best_params.items():
    print(f"  {k}: {v}")



%%time
# Train optimized XGBoost (GPU)
xgb_opt = XGBClassifier(**study_xgb.best_params, random_state=42, tree_method='hist', device='cuda', verbosity=0)
xgb_opt.fit(X_train, y_train)
xgb_opt_pred = xgb_opt.predict(X_val)
xgb_opt_acc = accuracy_score(y_val, xgb_opt_pred)

print(f"Optimized XGBoost (GPU): {xgb_opt_acc:.4f} (before: {xgb_acc:.4f})")
print(f"Improvement: {(xgb_opt_acc - xgb_acc)*100:.2f}%")



%%time
# Hyperparameter tuning for LightGBM with Optuna (GPU)
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'device': 'gpu',
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=30, show_progress_bar=True)

print(f"Best LightGBM accuracy (CV): {study_lgbm.best_value:.4f}")



# Best LightGBM parameters found
print("Best LightGBM params:")
for k, v in study_lgbm.best_params.items():
    print(f"  {k}: {v}")



%%time
# Train optimized LightGBM (GPU)
lgbm_opt = LGBMClassifier(**study_lgbm.best_params, random_state=42, device='gpu', verbose=-1)
lgbm_opt.fit(X_train, y_train)
lgbm_opt_pred = lgbm_opt.predict(X_val)
lgbm_opt_acc = accuracy_score(y_val, lgbm_opt_pred)

print(f"Optimized LightGBM (GPU): {lgbm_opt_acc:.4f} (before: {lgbm_acc:.4f})")
print(f"Improvement: {(lgbm_opt_acc - lgbm_acc)*100:.2f}%")



%%time
# CatBoost tuning - ANTI-OVERFITTING (GPU)
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'depth': trial.suggest_int('depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0',
        'verbose': 0
    }
    
    # Manual CV for CatBoost
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_vl, model.predict(X_vl)))
    return np.mean(scores)

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=True)

print(f"Best CatBoost accuracy (CV-5): {study_cat.best_value:.4f}")



# Best CatBoost parameters found
print("Best CatBoost params:")
for k, v in study_cat.best_params.items():
    print(f"  {k}: {v}")



%%time
cat_opt = CatBoostClassifier(**study_cat.best_params, random_seed=42, task_type='GPU', devices='0', verbose=0)
cat_opt.fit(X_train, y_train)
cat_opt_pred = cat_opt.predict(X_val)
cat_opt_acc = accuracy_score(y_val, cat_opt_pred)

print(f"Optimized CatBoost (GPU): {cat_opt_acc:.4f} (before: {cat_acc:.4f})")
print(f"Improvement: {(cat_opt_acc - cat_acc)*100:.2f}%")



# Final comparison of all tuned models
final_results = pd.DataFrame({
    'Model': ['XGBoost (tuned)', 'LightGBM (tuned)', 'CatBoost (tuned)'],
    'Accuracy': [xgb_opt_acc, lgbm_opt_acc, cat_opt_acc]
}).sort_values('Accuracy', ascending=False)

print("=" * 50)
print("FINAL COMPARISON (GPU)")
print("=" * 50)
for _, row in final_results.iterrows():
    marker = " ★" if row['Accuracy'] == final_results['Accuracy'].max() else ""
    print(f"  {row['Model']}: {row['Accuracy']:.4f}{marker}")
print("=" * 50)

best_model_name = final_results.iloc[0]['Model']
best_acc = final_results.iloc[0]['Accuracy']
print(f"\nBest: {best_model_name} ({best_acc:.4f})")



# Kaggle Submission - Train best model on full data
X_full = df_train[features_tree]
y_full = df_train['diagnosed_diabetes']
X_test = df_test[features_tree]

# Select and train best model
if 'CatBoost' in best_model_name:
    final_model = CatBoostClassifier(**study_cat.best_params, random_seed=42, task_type='GPU', devices='0', verbose=0)
elif 'LightGBM' in best_model_name:
    final_model = LGBMClassifier(**study_lgbm.best_params, random_state=42, device='gpu', verbose=-1)
else:
    final_model = XGBClassifier(**study_xgb.best_params, random_state=42, tree_method='hist', device='cuda', verbosity=0)

final_model.fit(X_full, y_full)
test_predictions = final_model.predict(X_test)

print(f"Model: {best_model_name}")
print(f"Trained on {len(X_full):,} samples")
print(f"Predictions: {len(test_predictions):,}")



# Create submission file
submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': test_predictions.astype(int)
})

submission.to_csv('submission.csv', index=False)

print(f"Submission saved to 'submission.csv'")
print(f"Shape: {submission.shape}")
submission.head(10)


