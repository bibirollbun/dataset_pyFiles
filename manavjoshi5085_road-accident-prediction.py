import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgbm
from catboost import CatBoostRegressor

# =====================================================================
# 1. LOAD DATA
# =====================================================================
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Store IDs and target
train_ids = train['id']
test_ids = test['id']
target = train['accident_risk']

# Drop ID and target
train = train.drop(['id', 'accident_risk'], axis=1)
test = test.drop(['id'], axis=1)

# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================
print("\nPerforming feature engineering...")

# Identify categorical and numerical columns
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical columns: {cat_cols}")
print(f"Numerical columns: {num_cols}")

# Combine train and test for consistent encoding
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined_original = combined.copy()

# Label encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined[col + '_le'] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le

# Frequency encoding
for col in cat_cols:
    freq = combined[col].value_counts(normalize=True).to_dict()
    combined[f'{col}_freq'] = combined[col].map(freq)

# Count encoding
for col in cat_cols:
    count = combined[col].value_counts().to_dict()
    combined[f'{col}_count'] = combined[col].map(count)

# Interaction features between numerical columns
if len(num_cols) >= 2:
    for i, col1 in enumerate(num_cols[:4]):
        for col2 in num_cols[i+1:5]:
            combined[f'{col1}_x_{col2}'] = combined[col1] * combined[col2]
            combined[f'{col1}_div_{col2}'] = combined[col1] / (combined[col2] + 1e-5)
            combined[f'{col1}_plus_{col2}'] = combined[col1] + combined[col2]
            combined[f'{col1}_minus_{col2}'] = combined[col1] - combined[col2]

# Statistical features for numerical columns
if len(num_cols) >= 2:
    combined['num_mean'] = combined[num_cols].mean(axis=1)
    combined['num_std'] = combined[num_cols].std(axis=1)
    combined['num_max'] = combined[num_cols].max(axis=1)
    combined['num_min'] = combined[num_cols].min(axis=1)
    combined['num_range'] = combined['num_max'] - combined['num_min']
    combined['num_median'] = combined[num_cols].median(axis=1)

# Polynomial features for top numerical columns
for col in num_cols[:3]:
    combined[f'{col}_squared'] = combined[col] ** 2
    combined[f'{col}_cubed'] = combined[col] ** 3
    combined[f'{col}_sqrt'] = np.sqrt(np.abs(combined[col]))
    combined[f'{col}_log'] = np.log1p(np.abs(combined[col]))

# Drop original categorical columns
combined = combined.drop(columns=cat_cols)

# Split back into train and test
train_fe = combined.iloc[:len(train)].reset_index(drop=True)
test_fe = combined.iloc[len(train):].reset_index(drop=True)

print(f"Features after engineering: {train_fe.shape[1]}")

# =====================================================================
# 3. MODEL DEFINITIONS
# =====================================================================
def get_models():
    """Define models with optimized hyperparameters"""
    models = {
        'lgbm': LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'xgb': XGBRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        ),
        'cat': CatBoostRegressor(
            iterations=2000,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=3,
            subsample=0.8,
            random_state=42,
            verbose=0,
            task_type='CPU'
        ),
        'rf': RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'et': ExtraTreesRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            min_samples_split=5,
            random_state=42
        )
    }
    return models

# =====================================================================
# 4. CROSS-VALIDATION
# =====================================================================
print("\nStarting cross-validation...")

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store out-of-fold predictions for stacking
num_models = 6
oof_predictions = np.zeros((len(train_fe), num_models))
test_predictions = np.zeros((len(test_fe), num_models))
model_scores = []

models = get_models()
model_names = list(models.keys())

for fold, (train_idx, val_idx) in enumerate(kf.split(train_fe)):
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/{n_folds}")
    print(f"{'='*60}")
    
    X_train = train_fe.iloc[train_idx].copy()
    X_val = train_fe.iloc[val_idx].copy()
    y_train = target.iloc[train_idx]
    y_val = target.iloc[val_idx]
    
    # Add target encoding for this fold (prevent leakage)
    for col in cat_cols:
        # Get original categorical values
        train_cat = combined_original.iloc[train_idx][col]
        val_cat = combined_original.iloc[val_idx][col]
        test_cat = combined_original.iloc[len(train):][col]
        
        # Calculate mean target for each category in training fold
        target_map = pd.DataFrame({
            'cat': train_cat,
            'target': y_train.values
        }).groupby('cat')['target'].mean().to_dict()
        
        # Map to train, validation, and test
        global_mean = y_train.mean()
        X_train[f'{col}_target'] = train_cat.map(target_map).fillna(global_mean).values
        X_val[f'{col}_target'] = val_cat.map(target_map).fillna(global_mean).values
        
        # For test set, create a copy to add target encoding
        if fold == 0:
            test_fe[f'{col}_target'] = 0  # Initialize
        test_fe[f'{col}_target'] += test_cat.map(target_map).fillna(global_mean).values / n_folds
    
    fold_scores = []
    
    for idx, (name, model) in enumerate(models.items()):
        print(f"\nTraining {name}...")
        
        # Train model with proper API for each library
        if name == 'lgbm':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgbm.early_stopping(stopping_rounds=100, verbose=False),
                    lgbm.log_evaluation(period=0)
                ]
            )
        elif name == 'xgb':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        elif name == 'cat':
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        # Predict validation set
        val_pred = model.predict(X_val)
        val_pred = np.clip(val_pred, 0, 1)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"{name} RMSE: {rmse:.5f}")
        fold_scores.append(rmse)
        
        # Store OOF predictions
        oof_predictions[val_idx, idx] = val_pred
        
        # Predict test set
        test_pred = model.predict(test_fe)
        test_pred = np.clip(test_pred, 0, 1)
        test_predictions[:, idx] += test_pred / n_folds
    
    model_scores.append(fold_scores)
    print(f"\nFold {fold + 1} Average RMSE: {np.mean(fold_scores):.5f}")

# =====================================================================
# 5. ENSEMBLE & RESULTS
# =====================================================================
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS")
print("="*60)

model_scores_array = np.array(model_scores)
for idx, name in enumerate(model_names):
    avg_score = model_scores_array[:, idx].mean()
    std_score = model_scores_array[:, idx].std()
    print(f"{name:10s}: {avg_score:.5f} (+/- {std_score:.5f})")

# Weighted average ensemble
weights = 1 / (model_scores_array.mean(axis=0) + 1e-5)
weights = weights / weights.sum()

print(f"\nOptimal weights: {dict(zip(model_names, weights))}")

# OOF ensemble prediction
oof_ensemble = np.average(oof_predictions, axis=1, weights=weights)
oof_ensemble = np.clip(oof_ensemble, 0, 1)
oof_rmse = np.sqrt(mean_squared_error(target, oof_ensemble))

print(f"\n{'='*60}")
print(f"OOF Ensemble RMSE: {oof_rmse:.5f}")
print(f"{'='*60}")

# Test ensemble prediction
test_ensemble = np.average(test_predictions, axis=1, weights=weights)
test_ensemble = np.clip(test_ensemble, 0, 1)

# =====================================================================
# 6. STACKING META-MODEL
# =====================================================================
print("\nTraining meta-model (stacking)...")

meta_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

meta_model.fit(oof_predictions, target)
meta_test_pred = meta_model.predict(test_predictions)
meta_test_pred = np.clip(meta_test_pred, 0, 1)

# Blend ensemble and meta predictions
final_predictions = 0.7 * test_ensemble + 0.3 * meta_test_pred
final_predictions = np.clip(final_predictions, 0, 1)

# =====================================================================
# 7. CREATE SUBMISSION
# =====================================================================
print("\nCreating submission file...")

submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(f"Predictions range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
print(f"Mean prediction: {final_predictions.mean():.4f}")

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Expected RMSE: ~{oof_rmse:.5f}")
print("Submission ready for upload!")




