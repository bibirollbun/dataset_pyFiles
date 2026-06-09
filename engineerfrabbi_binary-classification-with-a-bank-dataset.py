import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')


# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")


# Load datasets
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
    print('âœ… Datasets loaded successfully.')
except FileNotFoundError:
    print('â�Œ Ensure train.csv, test.csv, and sample_submission.csv are in the current directory.')
    exit()

print(f'ğŸ“Š Train data shape: {train_df.shape}')
print(f'ğŸ“Š Test data shape: {test_df.shape}')
print(f'ğŸ“Š Sample submission shape: {sample_submission_df.shape}')

# Quick data inspection
print('\n--- Train Data Info ---')
print(train_df.info())

print('\n--- Target Distribution ---')
target_dist = train_df['y'].value_counts(normalize=True)
print(target_dist)


# Check for missing values
print('\n--- Missing Values Check ---')
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()
print(f"Train missing values: {missing_train.sum()}")
print(f"Test missing values: {missing_test.sum()}")


# Identify feature types
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
categorical_features = train_df.select_dtypes(include='object').columns.tolist()


# Remove 'id' and 'y' from features
if 'id' in numerical_features:
    numerical_features.remove('id')
if 'y' in numerical_features:
    numerical_features.remove('y')


print(f'ğŸ“ˆ Numerical features ({len(numerical_features)}): {numerical_features}')
print(f'ğŸ“‹ Categorical features ({len(categorical_features)}): {categorical_features}')


for col in numerical_features:
    plt.figure(figsize=(10, 4))
    sns.histplot(train_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

    # Box plot to check for outliers and distribution across target classes
    plt.figure(figsize=(10, 4))
    sns.boxplot(x='y', y=col, data=train_df)
    plt.title(f'{col} by Target Variable')
    plt.show()


for col in categorical_features:
    plt.figure(figsize=(12, 5))
    sns.countplot(y=col, data=train_df, order = train_df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.show()

    # Proportion of 'y' for each category
    if 'y' in train_df.columns:
        category_y_prop = train_df.groupby(col)['y'].value_counts(normalize=True).unstack()
        print(f'\nProportion of y for {col}:\n{category_y_prop}')
        category_y_prop.plot(kind='bar', stacked=True, figsize=(12, 5))
        plt.title(f'Subscription Rate by {col}')
        plt.ylabel('Proportion')
        plt.show()


plt.figure(figsize=(12, 10))
sns.heatmap(train_df[numerical_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


print(f'ğŸ“ˆ Numerical features ({len(numerical_features)}): {numerical_features}')
print(f'ğŸ“‹ Categorical features ({len(categorical_features)}): {categorical_features}')


# Quick feature engineering
def create_features(df):
    """Create new features for better model performance"""
    df = df.copy()
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 40, 60, 100], 
                            labels=['young', 'adult', 'middle', 'senior'])
    
    # Duration categories
    if 'duration' in df.columns:
        df['duration_category'] = pd.cut(df['duration'], bins=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    
    # Campaign intensity
    if 'campaign' in df.columns:
        df['campaign_intensity'] = pd.cut(df['campaign'], bins=[0, 1, 3, 6, 100], 
                                        labels=['low', 'medium', 'high', 'very_high'])
    
    # Previous outcome interaction
    if 'poutcome' in df.columns and 'previous' in df.columns:
        df['prev_success'] = ((df['poutcome'] == 'success') & (df['previous'] > 0)).astype(int)
    
    return df


# Apply feature engineering
train_df = create_features(train_df)
test_df = create_features(test_df)

# Update feature lists
categorical_features = train_df.select_dtypes(include='object').columns.tolist()
if 'y' in categorical_features:
    categorical_features.remove('y')

print(f'ğŸ“Š Updated categorical features: {len(categorical_features)}')


# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)


# Prepare data
X_train = train_df.drop(['id', 'y'], axis=1)
y_train = train_df['y']
X_test = test_df.drop('id', axis=1)


# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print(f'âœ… Preprocessing complete!')
print(f'ğŸ“Š Processed training shape: {X_train_processed.shape}')
print(f'ğŸ“Š Processed test shape: {X_test_processed.shape}')


# Setup cross-validation
N_SPLITS = 5
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


# Early stopping configuration
EARLY_STOPPING_CONFIG = {
    'patience': 20,           # Stop if no improvement for 20 rounds
    'min_improvement': 0.001, # Minimum improvement threshold
    'max_trials': 25,         # Reduced from 50 for faster optimization
    'timeout': 300           # 5 minutes max per model optimization
}


def early_stopping_callback(study, trial):
    """Custom early stopping for Optuna trials"""
    if len(study.trials) >= EARLY_STOPPING_CONFIG['max_trials']:
        study.stop()
    
    # Stop if we haven't improved in the last 10 trials
    if len(study.trials) > 10:
        recent_values = [t.value for t in study.trials[-10:] if t.value is not None]
        if recent_values and max(recent_values) - min(recent_values) < EARLY_STOPPING_CONFIG['min_improvement']:
            print(f"ğŸ›‘ Early stopping: No significant improvement in last 10 trials")
            study.stop()


def objective_lgb_optimized(trial):
    """Optimized LightGBM objective with early stopping"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'early_stopping_rounds': EARLY_STOPPING_CONFIG['patience'],
        
        # Optimized parameter ranges
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 200),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train)):
        X_train_fold = X_train_processed[train_idx]
        X_val_fold = X_train_processed[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_CONFIG['patience'], verbose=False)]
        )
        
        val_pred = model.predict_proba(X_val_fold)[:, 1]
        fold_score = roc_auc_score(y_val_fold, val_pred)
        scores.append(fold_score)
        
        # Early stopping if performance is poor
        if fold_score < 0.7:  # Minimum acceptable performance
            return 0.7  # Return poor score to prune this trial
    
    return np.mean(scores)

print("ğŸš€ Starting LightGBM optimization with early stopping...")
study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_lgb.optimize(
    objective_lgb_optimized, 
    n_trials=EARLY_STOPPING_CONFIG['max_trials'],
    timeout=EARLY_STOPPING_CONFIG['timeout'],
    callbacks=[early_stopping_callback]
)

print(f'âœ… LightGBM Best ROC AUC: {study_lgb.best_value:.5f}')
print(f'ğŸ“‹ Best params: {study_lgb.best_params}')


def objective_xgb_optimized(trial):
    """Optimized XGBoost objective with early stopping"""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'early_stopping_rounds': EARLY_STOPPING_CONFIG['patience'],
        
        # Optimized parameter ranges
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train)):
        X_train_fold = X_train_processed[train_idx]
        X_val_fold = X_train_processed[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        val_pred = model.predict_proba(X_val_fold)[:, 1]
        fold_score = roc_auc_score(y_val_fold, val_pred)
        scores.append(fold_score)
        
        if fold_score < 0.7:
            return 0.7
    
    return np.mean(scores)

print("ğŸš€ Starting XGBoost optimization with early stopping...")
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_xgb.optimize(
    objective_xgb_optimized,
    n_trials=EARLY_STOPPING_CONFIG['max_trials'],
    timeout=EARLY_STOPPING_CONFIG['timeout'],
    callbacks=[early_stopping_callback]
)

print(f'âœ… XGBoost Best ROC AUC: {study_xgb.best_value:.5f}')
print(f'ğŸ“‹ Best params: {study_xgb.best_params}')


def objective_cat_optimized(trial):
    """Optimized CatBoost objective with early stopping"""
    params = {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': False,
        'random_seed': RANDOM_STATE,
        'early_stopping_rounds': EARLY_STOPPING_CONFIG['patience'],
        
        # Optimized parameter ranges
        'iterations': trial.suggest_int('iterations', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'depth': trial.suggest_int('depth', 4, 9),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 128, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    }
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train)):
        X_train_fold = X_train_processed[train_idx]
        X_val_fold = X_train_processed[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            use_best_model=True
        )
        
        val_pred = model.predict_proba(X_val_fold)[:, 1]
        fold_score = roc_auc_score(y_val_fold, val_pred)
        scores.append(fold_score)
        
        if fold_score < 0.7:
            return 0.7
    
    return np.mean(scores)

print("ğŸš€ Starting CatBoost optimization with early stopping...")
study_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_cat.optimize(
    objective_cat_optimized,
    n_trials=EARLY_STOPPING_CONFIG['max_trials'],
    timeout=EARLY_STOPPING_CONFIG['timeout'],
    callbacks=[early_stopping_callback]
)

print(f'âœ… CatBoost Best ROC AUC: {study_cat.best_value:.5f}')
print(f'ğŸ“‹ Best params: {study_cat.best_params}')


def objective_lr_fast(trial):
    """Fast Logistic Regression optimization"""
    params = {
        'C': trial.suggest_float('C', 1e-3, 100, log=True),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train)):
        X_train_fold = X_train_processed[train_idx]
        X_val_fold = X_train_processed[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        model = LogisticRegression(**params)
        model.fit(X_train_fold, y_train_fold)
        
        val_pred = model.predict_proba(X_val_fold)[:, 1]
        fold_score = roc_auc_score(y_val_fold, val_pred)
        scores.append(fold_score)
    
    return np.mean(scores)

print("ğŸš€ Starting Logistic Regression optimization...")
study_lr = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_lr.optimize(objective_lr_fast, n_trials=15, timeout=120)

print(f'âœ… Logistic Regression Best ROC AUC: {study_lr.best_value:.5f}')


# Train best models
print("\nğŸ�¯ Training final models with best parameters...")

# Best models with optimized parameters
best_models = {
    'LightGBM': lgb.LGBMClassifier(**study_lgb.best_params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
    'XGBoost': xgb.XGBClassifier(**study_xgb.best_params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    'CatBoost': CatBoostClassifier(**study_cat.best_params, random_seed=RANDOM_STATE, verbose=False),
    'LogisticRegression': LogisticRegression(**study_lr.best_params, random_state=RANDOM_STATE, n_jobs=-1)
}

# Performance tracking
model_scores = {}
test_predictions = {}
oof_predictions = {}

for name, model in best_models.items():
    print(f"\nğŸ”„ Training {name}...")
    
    # Out-of-fold predictions
    oof_pred = np.zeros(len(X_train_processed))
    test_pred = np.zeros(len(X_test_processed))
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train)):
        X_train_fold = X_train_processed[train_idx]
        X_val_fold = X_train_processed[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        # Train with early stopping for tree models
        if name in ['LightGBM', 'XGBoost', 'CatBoost']:
            if name == 'LightGBM':
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    callbacks=[lgb.early_stopping(20, verbose=False)]
                )
            elif name == 'XGBoost':
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
            else:  # CatBoost
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    use_best_model=True
                )
        else:
            model.fit(X_train_fold, y_train_fold)
        
        # Predictions
        val_pred = model.predict_proba(X_val_fold)[:, 1]
        oof_pred[val_idx] = val_pred
        test_pred += model.predict_proba(X_test_processed)[:, 1] / N_SPLITS
        
        fold_score = roc_auc_score(y_val_fold, val_pred)
        fold_scores.append(fold_score)
    
    # Store results
    final_score = roc_auc_score(y_train, oof_pred)
    model_scores[name] = {
        'OOF_Score': final_score,
        'Fold_Scores': fold_scores,
        'Std': np.std(fold_scores)
    }
    
    oof_predictions[name] = oof_pred
    test_predictions[name] = test_pred
    
    print(f"âœ… {name} - OOF Score: {final_score:.5f} (Â±{np.std(fold_scores):.5f})")


print("\n" + "="*60)
print("ğŸ“Š FINAL MODEL PERFORMANCE SUMMARY")
print("="*60)

performance_df = pd.DataFrame([
    {
        'Model': name,
        'OOF_ROC_AUC': scores['OOF_Score'],
        'CV_Std': scores['Std'],
        'Fold_Scores': ', '.join([f"{s:.4f}" for s in scores['Fold_Scores']])
    }
    for name, scores in model_scores.items()
]).sort_values('OOF_ROC_AUC', ascending=False)

print(performance_df.to_string(index=False))

# Create weighted ensemble based on performance
weights = {}
total_score = sum(scores['OOF_Score'] for scores in model_scores.values())
for name, scores in model_scores.items():
    weights[name] = scores['OOF_Score'] / total_score

print(f"\nğŸ�¯ Ensemble Weights: {weights}")

# Generate ensemble predictions
ensemble_test_pred = np.zeros(len(X_test_processed))
for name, weight in weights.items():
    ensemble_test_pred += weight * test_predictions[name]

print(f"âœ… Ensemble prediction created!")


submission = pd.DataFrame({
    'id': test_df['id'],
    'y': ensemble_test_pred
})


submission.to_csv('submission.csv', index=False)


print(f"\nğŸ�‰ Submission file created: submission.csv")
print(f"ğŸ“Š Prediction range: [{ensemble_test_pred.min():.4f}, {ensemble_test_pred.max():.4f}]")
print(f"ğŸ“Š Mean prediction: {ensemble_test_pred.mean():.4f}")


# Display first few predictions
print(f"\nğŸ“‹ First 10 predictions:")
print(submission.head(10).to_string(index=False))

print(f"\nğŸ�� Pipeline completed successfully!")
print(f"â�±ï¸�  Total optimization trials: LGB({len(study_lgb.trials)}), XGB({len(study_xgb.trials)}), CAT({len(study_cat.trials)}), LR({len(study_lr.trials)})")


# Optional: Quick feature importance for best LightGBM model
if 'LightGBM' in best_models:
    print(f"\nğŸ“ˆ Top 10 Important Features (LightGBM):")
    lgb_model = best_models['LightGBM']
    
    # Retrain on full data to get feature importance
    lgb_model.fit(X_train_processed, y_train)
    feature_names = (numerical_features + 
                    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    print(feature_importance.to_string(index=False))




