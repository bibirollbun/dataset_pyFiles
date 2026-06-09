import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import catboost as cb
import optuna
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

print("Loading data...")
path = '/kaggle/input/agriyield-2025/'
train = pd.read_csv(f'{path}train.csv')
test = pd.read_csv(f'{path}test.csv')
sample_submission = pd.read_csv(f'{path}sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

print("\nTrain data info:")
print(train.info())
print("\nTrain data description:")
print(train.describe())

print("\nMissing values in train:")
print(train.isnull().sum())
print("\nMissing values in test:")
print(test.isnull().sum())

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(train['yield'], bins=50, alpha=0.7, edgecolor='black')
plt.title('Target Distribution (Yield)')
plt.xlabel('Yield (kg/ha)')
plt.ylabel('Frequency')

plt.subplot(2, 3, 2)
corr_matrix = train.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')

features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi']
for i, feature in enumerate(features[:4]):
    plt.subplot(2, 3, i+3)
    plt.hist(train[feature], bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

def create_features(df):
    """Create additional features"""
    df = df.copy()
    
    df['ph_organic_interaction'] = df['soil_ph'] * df['organic_matter']
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    df['rainfall_ndvi_interaction'] = df['rainfall'] * df['ndvi']
    
    df['sand_clay_ratio'] = df['sand_pct'] / (100 - df['sand_pct'] - df.get('silt_pct', 0) + 1e-8)
    df['organic_ph_ratio'] = df['organic_matter'] / (df['soil_ph'] + 1e-8)
    
    df['ndvi_squared'] = df['ndvi'] ** 2
    df['temperature_squared'] = df['temperature'] ** 2
    df['rainfall_squared'] = df['rainfall'] ** 2
    
    df['rainfall_bin'] = pd.cut(df['rainfall'], bins=5, labels=False)
    df['temperature_bin'] = pd.cut(df['temperature'], bins=5, labels=False)
    df['ndvi_bin'] = pd.cut(df['ndvi'], bins=5, labels=False)
    
    return df

print("Creating features...")
train_fe = create_features(train)
test_fe = create_features(test)

feature_cols = [col for col in train_fe.columns if col not in ['field_id', 'yield']]
X = train_fe[feature_cols]
y = train_fe['yield']
X_test = test_fe[feature_cols]

print(f"Final feature count: {len(feature_cols)}")
print(f"Features: {feature_cols}")

N_FOLDS = 10
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def find_best_lr(model_type, X_train, y_train, X_val, y_val):
    learning_rates = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
    best_lr = 0.1
    best_score = float('inf')
    
    for lr in learning_rates:
        if model_type == 'xgb':
            model = xgb.XGBRegressor(
                learning_rate=lr,
                n_estimators=100,
                random_state=SEED,
                tree_method='gpu_hist',
                verbosity=0
            )
        elif model_type == 'cat':
            model = cb.CatBoostRegressor(
                learning_rate=lr,
                iterations=100,
                random_seed=SEED,
                task_type='GPU',
                verbose=False
            )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, preds))
        
        if score < best_score:
            best_score = score
            best_lr = lr
    
    print(f"Best learning rate for {model_type}: {best_lr} (RMSE: {best_score:.4f})")
    return best_lr


def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': 1000,
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0),
        'tree_method': 'gpu_hist',
        'random_state': SEED,
        'verbosity': 0
    }
    
    cv_scores = []
    for train_idx, val_idx in kfold.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  early_stopping_rounds=50, verbose=False)
        
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        cv_scores.append(rmse)
    
    return np.mean(cv_scores)

def objective_cat(trial):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'task_type': 'GPU',
        'random_seed': SEED,
        'verbose': False
    }
    
    cv_scores = []
    for train_idx, val_idx in kfold.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=(X_val_fold, y_val_fold),
                  early_stopping_rounds=50, verbose=False)
        
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        cv_scores.append(rmse)
    
    return np.mean(cv_scores)

print("Starting hyperparameter optimization...")

print("Optimizing XGBoost...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=50)
best_params_xgb = study_xgb.best_params
print(f"Best XGBoost params: {best_params_xgb}")

print("Optimizing CatBoost...")
study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=50)
best_params_cat = study_cat.best_params
print(f"Best CatBoost params: {best_params_cat}")

def train_model_cv(model_type, params):
    """Train model using cross-validation"""
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"Training {model_type} - Fold {fold + 1}/{N_FOLDS}")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]        
        
        if model_type == 'xgb':
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      early_stopping_rounds=100, verbose=False)
        
        elif model_type == 'cat':
            model = cb.CatBoostRegressor(**params)
            model.fit(X_train_fold, y_train_fold,
                      eval_set=(X_val_fold, y_val_fold),
                      early_stopping_rounds=100, verbose=False)
        
        oof_preds[val_idx] = model.predict(X_val_fold)
        test_preds += model.predict(X_test) / N_FOLDS
        
        rmse = np.sqrt(mean_squared_error(y_val_fold, oof_preds[val_idx]))
        cv_scores.append(rmse)
        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")
    
    cv_score = np.mean(cv_scores)
    print(f"{model_type} CV RMSE: {cv_score:.4f} ± {np.std(cv_scores):.4f}")
    
    return oof_preds, test_preds, cv_score

print("\nTraining models with optimized parameters...")

best_params_xgb.update({'tree_method': 'gpu_hist', 'random_state': SEED, 'verbosity': 0})
oof_xgb, test_xgb, cv_xgb = train_model_cv('xgb', best_params_xgb)

best_params_cat.update({'task_type': 'GPU', 'random_seed': SEED, 'verbose': False})
oof_cat, test_cat, cv_cat = train_model_cv('cat', best_params_cat)

# Meta-learner optimization
print("\nOptimizing meta-learner...")

def objective_meta(trial):
    """Optimize meta-learner using out-of-fold predictions"""
    meta_X = np.column_stack([ oof_xgb, oof_cat])
    
    meta_type = trial.suggest_categorical('meta_type', ['ridge', 'lasso', 'elastic', 'rf'])
    
    if meta_type == 'ridge':
        alpha = trial.suggest_float('alpha', 1e-5, 10.0)
        meta_model = Ridge(alpha=alpha, random_state=SEED)
    elif meta_type == 'lasso':
        alpha = trial.suggest_float('alpha', 1e-5, 1.0)
        meta_model = Lasso(alpha=alpha, random_state=SEED)
    elif meta_type == 'elastic':
        alpha = trial.suggest_float('alpha', 1e-5, 1.0)
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
        meta_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=SEED)
    else:  # random forest
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        meta_model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=SEED
        )
    
    cv_scores = []
    for train_idx, val_idx in kfold.split(meta_X, y):
        meta_X_train, meta_X_val = meta_X[train_idx], meta_X[val_idx]
        meta_y_train, meta_y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        meta_model.fit(meta_X_train, meta_y_train)
        meta_preds = meta_model.predict(meta_X_val)
        rmse = np.sqrt(mean_squared_error(meta_y_val, meta_preds))
        cv_scores.append(rmse)
    
    return np.mean(cv_scores)

study_meta = optuna.create_study(direction='minimize')
study_meta.optimize(objective_meta, n_trials=60)
best_meta_params = study_meta.best_params
print(f"Best meta-learner params: {best_meta_params}")

meta_X = np.column_stack([oof_xgb, oof_cat])
meta_test_X = np.column_stack([test_xgb, test_cat])

scaler = StandardScaler()
meta_X_scaled = scaler.fit_transform(meta_X)
meta_test_X_scaled = scaler.transform(meta_test_X)

if best_meta_params['meta_type'] == 'ridge':
    final_meta = Ridge(alpha=best_meta_params['alpha'], random_state=SEED)
elif best_meta_params['meta_type'] == 'lasso':
    final_meta = Lasso(alpha=best_meta_params['alpha'], random_state=SEED)
elif best_meta_params['meta_type'] == 'elastic':
    final_meta = ElasticNet(
        alpha=best_meta_params['alpha'],
        l1_ratio=best_meta_params['l1_ratio'],
        random_state=SEED
    )
else:
    final_meta = RandomForestRegressor(
        n_estimators=best_meta_params['n_estimators'],
        max_depth=best_meta_params['max_depth'],
        random_state=SEED
    )

final_meta.fit(meta_X_scaled, y)
meta_cv_score = np.sqrt(mean_squared_error(y, final_meta.predict(meta_X)))
print(f"Meta-learner CV RMSE: {meta_cv_score:.4f}")

final_test_preds = final_meta.predict(meta_test_X_scaled)

print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"XGBoost CV RMSE:  {cv_xgb:.4f}")
print(f"CatBoost CV RMSE: {cv_cat:.4f}")
print(f"Ensemble CV RMSE: {meta_cv_score:.4f}")

simple_ensemble = (test_xgb + test_cat) / 3

print("\nAnalyzing feature importance...")

xgb_final = xgb.XGBRegressor(**best_params_xgb)
xgb_final.fit(X, y)

cat_final = cb.CatBoostRegressor(**best_params_cat)
cat_final.fit(X, y, verbose=False)

xgb_importance = xgb_final.feature_importances_
cat_importance = cat_final.feature_importances_

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'xgb_importance': xgb_importance,
    'cat_importance': cat_importance
})

importance_df['avg_importance'] = (#importance_df['lgb_importance'] + 
                                  importance_df['xgb_importance'] + 
                                  importance_df['cat_importance']) / 3

importance_df = importance_df.sort_values('avg_importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10)[['feature', 'avg_importance']])

# Visualization
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)

plt.subplot(1, 2, 1)
plt.barh(range(len(top_features)), top_features['avg_importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Average Importance')
plt.title('Top 15 Features by Average Importance')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.scatter(y, final_meta.predict(meta_X), alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title(f'Actual vs Predicted (RMSE: {meta_cv_score:.4f})')

plt.tight_layout()
plt.show()

# Create submissions
print("\nCreating submission files...")

# Ensemble submission
submission_ensemble = sample_submission.copy()
submission_ensemble['yield'] = final_test_preds
submission_ensemble.to_csv('submission_ensemble.csv', index=False)

# Simple ensemble backup
submission_simple = sample_submission.copy()
submission_simple['yield'] = simple_ensemble
submission_simple.to_csv('submission_simple.csv', index=False)

print("\nSubmission files created:")
print("- submission_ensemble.csv (Meta-learner ensemble)")
print("- submission_simple.csv (Simple average)")
# print("- submission_lgb.csv (LightGBM only)")

print(f"\nFinal ensemble predictions range: [{final_test_preds.min():.2f}, {final_test_preds.max():.2f}]")
print(f"Training target range: [{y.min():.2f}, {y.max():.2f}]")

print("\n" + "="*50)
print("NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
print("="*50)


oof_predictions_list = [oof_xgb, oof_cat]
y_true = y
def rmse_optimizer(weights):
    """Function to minimize. It calculates the RMSE of a weighted average."""
    final_prediction = 0
    for weight, prediction in zip(weights, oof_predictions_list):
        final_prediction += weight * prediction
    return np.sqrt(mean_squared_error(y, final_prediction))

# Initial guess (equal weights)
initial_weights = [1/len(oof_predictions_list)] * len(oof_predictions_list)

# Constraint: weights must sum to 1
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

# Bounds: weights must be between 0 and 1
bounds = [(0, 1)] * len(oof_predictions_list)

# Run the optimization
result = minimize(rmse_optimizer,
                  initial_weights,
                  method='SLSQP',
                  bounds=bounds,
                  constraints=constraints)

# Get the best weights
best_weights = result['x']
optimized_rmse = result['fun']

print("-" * 50)
print(f"Optimized RMSE on OOF predictions: {optimized_rmse:.4f}")
print(f"Best Weights (XGB, CatBoost): {[f'{w:.4f}' for w in best_weights]}")
print("-" * 50)

# Create final predictions using the optimized weights
weighted_test_preds = (best_weights[0] * test_xgb +
                       best_weights[1] * test_cat)

# Create a submission file for this too, it might be your best one!
submission_weighted = sample_submission.copy()
submission_weighted['yield'] = weighted_test_preds
submission_weighted.to_csv('submission_weighted_avg.csv', index=False)

