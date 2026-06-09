import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

# Model imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load data
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# Store IDs for submission
test_ids = test['id']

# Separate features and target
X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']
X_test = test.drop('id', axis=1)


print("\n" + "="*50)
print("Advanced Data Analysis")
print("="*50)

print(f"\nTarget distribution:")
print(y.value_counts())
print(f"Class imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.2f}:1")


# Check for missing values
missing_values = X.isnull().sum()
print(f"\nMissing values in training data:")
print(missing_values[missing_values > 0].sort_values(ascending=False))


# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")


def advanced_preprocessing(X_train, X_test, categorical_cols, numerical_cols, target=None):
    """
    Advanced preprocessing with feature engineering and outlier handling
    """
    # Create copies
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Store transformations
    transformers = {}
    
    # ============================================
    # 1. Handle Missing Values
    # ============================================
    print("Step 1: Handling missing values...")
    
    # For numerical columns: use different strategies based on distribution
    num_imputer = SimpleImputer(strategy='median')
    X_train_processed[numerical_cols] = num_imputer.fit_transform(X_train_processed[numerical_cols])
    X_test_processed[numerical_cols] = num_imputer.transform(X_test_processed[numerical_cols])
    transformers['num_imputer'] = num_imputer
    
    # For categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_processed[categorical_cols] = cat_imputer.fit_transform(X_train_processed[categorical_cols])
    X_test_processed[categorical_cols] = cat_imputer.transform(X_test_processed[categorical_cols])
    transformers['cat_imputer'] = cat_imputer
    
    # ============================================
    # 2. Handle Outliers in Numerical Features
    # ============================================
    print("Step 2: Handling outliers...")
    
    def cap_outliers(series, lower_quantile=0.01, upper_quantile=0.99):
        lower_bound = series.quantile(lower_quantile)
        upper_bound = series.quantile(upper_quantile)
        return series.clip(lower_bound, upper_bound)
    
    for col in numerical_cols:
        lower_bound = X_train_processed[col].quantile(0.01)
        upper_bound = X_train_processed[col].quantile(0.99)
        X_train_processed[col] = X_train_processed[col].clip(lower_bound, upper_bound)
        X_test_processed[col] = X_test_processed[col].clip(lower_bound, upper_bound)
    
    # ============================================
    # 3. Encode Categorical Variables
    # ============================================
    print("Step 3: Encoding categorical variables...")
    encoders = {}
    
    for col in categorical_cols:
        # Combine train and test for encoding
        combined = pd.concat([X_train_processed[col], X_test_processed[col]], axis=0).astype(str)
        
        # Use frequency encoding for high cardinality features
        freq_encoding = combined.value_counts(normalize=True).to_dict()
        X_train_processed[f'{col}_freq'] = X_train_processed[col].astype(str).map(freq_encoding)
        X_test_processed[f'{col}_freq'] = X_test_processed[col].astype(str).map(freq_encoding)
        
        # Use LabelEncoder for tree-based models
        le = LabelEncoder()
        le.fit(combined.fillna('Missing'))
        X_train_processed[f'{col}_encoded'] = le.transform(X_train_processed[col].astype(str).fillna('Missing'))
        X_test_processed[f'{col}_encoded'] = le.transform(X_test_processed[col].astype(str).fillna('Missing'))
        encoders[col] = le
    
    # Remove original categorical columns
    X_train_processed = X_train_processed.drop(categorical_cols, axis=1)
    X_test_processed = X_test_processed.drop(categorical_cols, axis=1)
    
    # ============================================
    # 4. Advanced Feature Engineering
    # ============================================
    print("Step 4: Advanced feature engineering...")
    
    # Medical risk scores
    X_train_processed['metabolic_syndrome_score'] = (
        (X_train_processed['bmi'] > 30).astype(int) +
        (X_train_processed['waist_to_hip_ratio'] > 0.9).astype(int) +
        (X_train_processed['triglycerides'] > 150).astype(int) +
        (X_train_processed['hdl_cholesterol'] < 40).astype(int) +
        (X_train_processed['systolic_bp'] > 130).astype(int)
    )
    
    X_test_processed['metabolic_syndrome_score'] = (
        (X_test_processed['bmi'] > 30).astype(int) +
        (X_test_processed['waist_to_hip_ratio'] > 0.9).astype(int) +
        (X_test_processed['triglycerides'] > 150).astype(int) +
        (X_test_processed['hdl_cholesterol'] < 40).astype(int) +
        (X_test_processed['systolic_bp'] > 130).astype(int)
    )
    
    # BMI categories with finer granularity
    bmi_bins = [0, 18.5, 23, 25, 30, 35, 40, 100]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese1', 'obese2', 'obese3', 'obese4']
    X_train_processed['bmi_category'] = pd.cut(X_train_processed['bmi'], bins=bmi_bins, labels=range(len(bmi_labels)))
    X_test_processed['bmi_category'] = pd.cut(X_test_processed['bmi'], bins=bmi_bins, labels=range(len(bmi_labels)))
    
    # Age-Lifestyle Interactions
    X_train_processed['age_activity'] = X_train_processed['age'] * X_train_processed['physical_activity_minutes_per_week']
    X_test_processed['age_activity'] = X_test_processed['age'] * X_test_processed['physical_activity_minutes_per_week']
    
    X_train_processed['bmi_cholesterol'] = X_train_processed['bmi'] * X_train_processed['cholesterol_total']
    X_test_processed['bmi_cholesterol'] = X_test_processed['bmi'] * X_test_processed['cholesterol_total']
    
    # Blood pressure status
    X_train_processed['hypertension_status'] = ((X_train_processed['systolic_bp'] >= 140) | 
                                                (X_train_processed['diastolic_bp'] >= 90)).astype(int)
    X_test_processed['hypertension_status'] = ((X_test_processed['systolic_bp'] >= 140) | 
                                               (X_test_processed['diastolic_bp'] >= 90)).astype(int)
    
    # Cholesterol ratios with log transformation
    X_train_processed['log_chol_hdl_ratio'] = np.log1p(X_train_processed['cholesterol_total'] / X_train_processed['hdl_cholesterol'])
    X_test_processed['log_chol_hdl_ratio'] = np.log1p(X_test_processed['cholesterol_total'] / X_test_processed['hdl_cholesterol'])
    
    # Triglyceride to HDL ratio (strong predictor)
    X_train_processed['tg_hdl_ratio'] = X_train_processed['triglycerides'] / X_train_processed['hdl_cholesterol']
    X_test_processed['tg_hdl_ratio'] = X_test_processed['triglycerides'] / X_test_processed['hdl_cholesterol']
    
    # Lifestyle composite score
    X_train_processed['lifestyle_score'] = (
        X_train_processed['diet_score'] / 10 +
        X_train_processed['sleep_hours_per_day'] / 8 -
        X_train_processed['screen_time_hours_per_day'] / 10 -
        X_train_processed['alcohol_consumption_per_week'] / 10
    )
    
    X_test_processed['lifestyle_score'] = (
        X_test_processed['diet_score'] / 10 +
        X_test_processed['sleep_hours_per_day'] / 8 -
        X_test_processed['screen_time_hours_per_day'] / 10 -
        X_test_processed['alcohol_consumption_per_week'] / 10
    )
    
    # Polynomial features for important variables
    for col in ['bmi', 'age', 'waist_to_hip_ratio', 'cholesterol_total']:
        X_train_processed[f'{col}_squared'] = X_train_processed[col] ** 2
        X_test_processed[f'{col}_squared'] = X_test_processed[col] ** 2
        X_train_processed[f'{col}_cubed'] = X_train_processed[col] ** 3
        X_test_processed[f'{col}_cubed'] = X_test_processed[col] ** 3
    
    # ============================================
    # 5. Feature Scaling and Transformation
    # ============================================
    print("Step 5: Feature scaling and transformation...")
    
    # Use RobustScaler for numerical features (less sensitive to outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    transformers['scaler'] = scaler
    
    # Apply PowerTransformer for skewed features
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_train_scaled = power_transformer.fit_transform(X_train_scaled)
    X_test_scaled = power_transformer.transform(X_test_scaled)
    
    transformers['power_transformer'] = power_transformer
    
    # Convert back to DataFrame
    X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train_processed.columns, index=X_train_processed.index)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=X_test_processed.columns, index=X_test_processed.index)
    
    # ============================================
    # 6. Feature Selection (optional)
    # ============================================
    if target is not None:
        print("Step 6: Feature selection...")
        # Select top k features based on ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k=min(50, X_train_processed.shape[1]))
        X_train_selected = selector.fit_transform(X_train_processed, target)
        X_test_selected = selector.transform(X_test_processed)
        
        selected_features = X_train_processed.columns[selector.get_support()]
        X_train_processed = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train_processed.index)
        X_test_processed = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test_processed.index)
        
        transformers['selector'] = selector
    
    return X_train_processed, X_test_processed, transformers, encoders


print("\n" + "="*50)
print("Applying Advanced Preprocessing")
print("="*50)

X_processed, X_test_processed, transformers, encoders = advanced_preprocessing(
    X, X_test, categorical_cols, numerical_cols, y
)

print(f"Processed training data shape: {X_processed.shape}")
print(f"Processed test data shape: {X_test_processed.shape}")


X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test_processed.shape}")


print("\n" + "="*50)
print("Hyperparameter Optimization with Optuna")
print("="*50)

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    model_name = trial.suggest_categorical('model', ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest'])
    
    if model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('xgb_gamma', 0, 5),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
            'scale_pos_weight': trial.suggest_float('xgb_scale_pos_weight', 1, 10),
        }
        model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='auc', n_jobs=-1)
    
    elif model_name == 'LightGBM':
        params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 100),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0, 10),
            'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
            'class_weight': 'balanced',
        }
        model = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbosity=-1, silent=True)
    
    elif model_name == 'CatBoost':
        params = {
            'iterations': trial.suggest_int('cat_iterations', 100, 1000),
            'depth': trial.suggest_int('cat_depth', 4, 10),
            'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('cat_border_count', 32, 255),
            'random_strength': trial.suggest_float('cat_random_strength', 0, 10),
            'bagging_temperature': trial.suggest_float('cat_bagging_temperature', 0, 1),
        }
        model = CatBoostClassifier(**params, random_state=42, verbose=0)
    
    elif model_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('rf_bootstrap', [True, False]),
            'class_weight': 'balanced',
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return cv_scores.mean()


# Create Optuna study
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

# Optimize
print("Running hyperparameter optimization...")
study.optimize(objective, n_trials=10, show_progress_bar=True)

print(f"\nBest trial:")
print(f"  Value (AUC): {study.best_value:.4f}")
print(f"  Params: {study.best_params}")


print("\n" + "="*50)
print("Building Optimized Models")
print("="*50)

best_params = study.best_params
best_model_name = best_params['model']

# Define optimized models
optimized_models = {}

if best_model_name == 'XGBoost':
    optimized_models['XGBoost'] = XGBClassifier(
        n_estimators=best_params['xgb_n_estimators'],
        max_depth=best_params['xgb_max_depth'],
        learning_rate=best_params['xgb_learning_rate'],
        subsample=best_params['xgb_subsample'],
        colsample_bytree=best_params['xgb_colsample_bytree'],
        gamma=best_params['xgb_gamma'],
        reg_alpha=best_params['xgb_reg_alpha'],
        reg_lambda=best_params['xgb_reg_lambda'],
        min_child_weight=best_params['xgb_min_child_weight'],
        scale_pos_weight=best_params['xgb_scale_pos_weight'],
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1
    )

elif best_model_name == 'LightGBM':
    optimized_models['LightGBM'] = LGBMClassifier(
        n_estimators=best_params['lgb_n_estimators'],
        max_depth=best_params['lgb_max_depth'],
        learning_rate=best_params['lgb_learning_rate'],
        num_leaves=best_params['lgb_num_leaves'],
        subsample=best_params['lgb_subsample'],
        colsample_bytree=best_params['lgb_colsample_bytree'],
        reg_alpha=best_params['lgb_reg_alpha'],
        reg_lambda=best_params['lgb_reg_lambda'],
        min_child_samples=best_params['lgb_min_child_samples'],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

elif best_model_name == 'CatBoost':
    optimized_models['CatBoost'] = CatBoostClassifier(
        iterations=best_params['cat_iterations'],
        depth=best_params['cat_depth'],
        learning_rate=best_params['cat_learning_rate'],
        l2_leaf_reg=best_params['cat_l2_leaf_reg'],
        border_count=best_params['cat_border_count'],
        random_strength=best_params['cat_random_strength'],
        bagging_temperature=best_params['cat_bagging_temperature'],
        random_state=42,
        verbose=0
    )

elif best_model_name == 'RandomForest':
    optimized_models['RandomForest'] = RandomForestClassifier(
        n_estimators=best_params['rf_n_estimators'],
        max_depth=best_params['rf_max_depth'],
        min_samples_split=best_params['rf_min_samples_split'],
        min_samples_leaf=best_params['rf_min_samples_leaf'],
        max_features=best_params['rf_max_features'],
        bootstrap=best_params['rf_bootstrap'],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

# Add other models with reasonable defaults
optimized_models['HistGradientBoosting'] = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)

optimized_models['AdaBoost'] = AdaBoostClassifier(
    n_estimators=300,
    learning_rate=0.1,
    random_state=42
)


print("\nTraining and evaluating optimized models...")

results = {}
cv_scores = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in optimized_models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    cv_auc_scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    cv_scores[name] = cv_auc_scores
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_val_pred = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_val, y_val_pred)
    average_precision = average_precision_score(y_val, y_val_pred)
    
    # Calculate optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    y_val_pred_class = (y_val_pred >= optimal_threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred_class)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results[name] = {
        'model': model,
        'auc': auc_score,
        'average_precision': average_precision,
        'cv_mean': cv_auc_scores.mean(),
        'cv_std': cv_auc_scores.std(),
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm
    }
    
    print(f"  Validation AUC: {auc_score:.4f}")
    print(f"  Average Precision: {average_precision:.4f}")
    print(f"  CV AUC: {cv_auc_scores.mean():.4f} (+/- {cv_auc_scores.std()*2:.4f})")
    print(f"  F1-Score: {f1_score:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")


print("\n" + "="*50)
print("Model Comparison")
print("="*50)

results_df = pd.DataFrame([
    {
        'Model': name,
        'Validation AUC': results[name]['auc'],
        'Average Precision': results[name]['average_precision'],
        'CV Mean AUC': results[name]['cv_mean'],
        'CV Std AUC': results[name]['cv_std'],
        'F1-Score': results[name]['f1_score'],
        'Accuracy': results[name]['accuracy'],
        'Precision': results[name]['precision'],
        'Recall': results[name]['recall'],
        'Optimal Threshold': results[name]['optimal_threshold']
    }
    for name in results.keys()
]).sort_values('Validation AUC', ascending=False)

print(results_df.to_string(index=False))


# Select best model based on AUC
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\nâœ¨ Best Model: {best_model_name}")
print(f"   Validation AUC: {results[best_model_name]['auc']:.4f}")
print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")


print("\n" + "="*50)
print("Creating Ensemble Model")
print("="*50)

# Select top 3 models for ensemble
top_n = 3
top_models = results_df.head(top_n)['Model'].tolist()

print(f"Creating ensemble from top {top_n} models: {top_models}")


# Create a voting classifier (soft voting for probabilities)
ensemble_models = [(name, results[name]['model']) for name in top_models]

voting_clf = VotingClassifier(
    estimators=ensemble_models,
    voting='soft',  # Use probabilities
    weights=[results[name]['auc'] for name in top_models]  # Weight by AUC
)

# Train ensemble model
voting_clf.fit(X_train, y_train)

# Evaluate ensemble
y_val_pred_ensemble = voting_clf.predict_proba(X_val)[:, 1]
ensemble_auc = roc_auc_score(y_val, y_val_pred_ensemble)
ensemble_ap = average_precision_score(y_val, y_val_pred_ensemble)

print(f"Ensemble Validation AUC: {ensemble_auc:.4f}")
print(f"Ensemble Average Precision: {ensemble_ap:.4f}")

# Compare with best single model
if ensemble_auc > results[best_model_name]['auc']:
    print("âœ… Ensemble performs better than best single model!")
    final_model = voting_clf
    final_model_name = f"Ensemble ({', '.join(top_models)})"
else:
    print("âœ… Best single model performs better than ensemble.")
    final_model = best_model
    final_model_name = best_model_name


print("\n" + "="*50)
print("Generating Advanced Visualizations")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Advanced Model Evaluation', fontsize=16, fontweight='bold')

# 1. Model Comparison Bar Plot
ax1 = axes[0, 0]
models_names = list(results.keys())
auc_scores = [results[name]['auc'] for name in models_names]
bars = ax1.barh(models_names, auc_scores, color=sns.color_palette("husl", len(models_names)))
ax1.set_xlabel('ROC AUC Score')
ax1.set_title('Model Comparison (AUC)')
ax1.axvline(x=max(auc_scores), color='red', linestyle='--', alpha=0.5, label='Best')

# Add value labels
for i, (bar, score) in enumerate(zip(bars, auc_scores)):
    ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.4f}', 
            va='center', ha='left', fontsize=9)

# 2. Cross-validation Scores Boxplot
ax2 = axes[0, 1]
cv_data = [cv_scores[name] for name in models_names]
box = ax2.boxplot(cv_data, labels=models_names, vert=False, patch_artist=True)
colors = sns.color_palette("husl", len(models_names))
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_xlabel('ROC AUC Score')
ax2.set_title('Cross-validation AUC Scores (5-fold)')

# 3. Precision-Recall Curve for Best Model
ax3 = axes[0, 2]
precision, recall, _ = precision_recall_curve(y_val, y_val_pred_ensemble)
ax3.plot(recall, precision, color='darkorange', lw=2, 
         label=f'AP = {ensemble_ap:.4f}')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title(f'Precision-Recall Curve - {final_model_name}')
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3)

# 4. ROC Curves Comparison
ax4 = axes[1, 0]
for name in top_models[:3]:  # Plot top 3 models
    y_val_pred = results[name]['model'].predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_val_pred)
    auc_score = results[name]['auc']
    ax4.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.4f})')

ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curves Comparison (Top 3 Models)')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# 5. Feature Importance (if available)
ax5 = axes[1, 1]
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax5, palette='viridis')
    ax5.set_xlabel('Importance')
    ax5.set_ylabel('Feature')
    ax5.set_title(f'Top 15 Feature Importance - {final_model_name}')
elif isinstance(final_model, VotingClassifier):
    # For ensemble, average feature importance from base models
    feature_importances = []
    for name, model in ensemble_models:
        if hasattr(model, 'feature_importances_'):
            feature_importances.append(model.feature_importances_)
    
    if feature_importances:
        avg_importance = np.mean(feature_importances, axis=0)
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': avg_importance
        }).sort_values('importance', ascending=False).head(15)
        
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax5, palette='viridis')
        ax5.set_xlabel('Average Importance')
        ax5.set_ylabel('Feature')
        ax5.set_title(f'Top 15 Feature Importance - Ensemble')

# 6. Calibration Curve (Probability Calibration)
ax6 = axes[1, 2]
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_val, y_val_pred_ensemble, n_bins=10)
ax6.plot(prob_pred, prob_true, 's-', label=f'{final_model_name}')
ax6.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax6.set_xlabel('Mean predicted probability')
ax6.set_ylabel('Fraction of positives')
ax6.set_title('Calibration Curve')
ax6.legend(loc='lower right')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()


print(f"\n=== Retraining {final_model_name} on Full Dataset ===")

# Retrain on full dataset
if final_model_name.startswith('Ensemble'):
    # For ensemble, retrain all base models on full data
    retrained_models = []
    for name, model in ensemble_models:
        model_class = type(model)
        model_params = model.get_params()
        if name == 'CatBoost':
            retrained_model = CatBoostClassifier(**model_params, verbose=0)
        else:
            retrained_model = model_class(**model_params)
        retrained_model.fit(X_processed, y)
        retrained_models.append((name, retrained_model))
    
    final_model = VotingClassifier(
        estimators=retrained_models,
        voting='soft',
        weights=[results[name]['auc'] for name in top_models]
    )
    final_model.fit(X_processed, y)
else:
    # For single model
    model_class = type(final_model)
    model_params = final_model.get_params()
    
    if final_model_name == 'CatBoost':
        final_model = CatBoostClassifier(**model_params, iterations=500, verbose=0)
    elif hasattr(final_model, 'n_estimators'):
        # Increase number of estimators for final model
        final_model.set_params(n_estimators=int(model_params.get('n_estimators', 300) * 1.2))
    elif hasattr(final_model, 'max_iter'):
        final_model.set_params(max_iter=int(model_params.get('max_iter', 100) * 1.2))
    
    final_model.fit(X_processed, y)

print("Training completed!")


print("\n=== Making Predictions on Test Set ===")
test_predictions = final_model.predict_proba(X_test_processed)[:, 1]

# Apply calibration if needed (optional)
# You can add probability calibration here if models are over/under confident

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_predictions
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print(f"\nâœ… Optimized submission file saved as 'submission_optimized.csv'")
print(f"Shape: {submission.shape}")
print(f"Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
print(f"Mean prediction: {test_predictions.mean():.4f}")
print(f"Std prediction: {test_predictions.std():.4f}")

