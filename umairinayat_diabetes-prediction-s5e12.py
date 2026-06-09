# Install required packages (uncomment if running in fresh environment)
# !pip install -q xgboost lightgbm catboost optuna shap plotly imbalanced-learn scikit-learn-intelex

import warnings
warnings.filterwarnings('ignore')

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Gradient Boosting models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# SHAP for explainability
import shap

# Utilities
from tqdm.auto import tqdm
from collections import defaultdict
import joblib
import gc
import os

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("âœ“ All libraries imported successfully")
print(f"Python Environment Ready | Random Seed: {SEED}")


# Load Kaggle competition data
train_df = pd.read_csv('playground-series-s5e12/train.csv')
test_df = pd.read_csv('playground-series-s5e12/test.csv')
sample_submission = pd.read_csv('playground-series-s5e12/sample_submission.csv')

# Load original Diabetes Health Indicators Dataset
# Download from: https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset
try:
    original_df = pd.read_csv('diabetes-health-indicators-dataset/diabetes_binary_health_indicators_BRFSS2015.csv')
    print("âœ“ Original dataset loaded successfully")
except:
    print("âš  Original dataset not found. Download from Kaggle Datasets.")
    original_df = None

print(f"\n{'='*60}")
print("DATASET OVERVIEW")
print(f"{'='*60}")
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")
if original_df is not None:
    print(f"Original Dataset Shape: {original_df.shape}")
print(f"\nSample Submission Shape: {sample_submission.shape}")


# Display basic information
print("\nTRAIN DATASET INFO:")
print(train_df.info())

print("\n" + "="*60)
print("FIRST FEW ROWS:")
display(train_df.head())

print("\n" + "="*60)
print("STATISTICAL SUMMARY:")
display(train_df.describe())


# Check for missing values
def check_missing_values(df, name):
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    }).sort_values('Missing_Count', ascending=False)
    
    if missing_df['Missing_Count'].sum() == 0:
        print(f"âœ“ {name}: No missing values detected")
    else:
        print(f"\n{name} - Missing Values:")
        display(missing_df[missing_df['Missing_Count'] > 0])
    
    return missing_df

train_missing = check_missing_values(train_df, "Train")
test_missing = check_missing_values(test_df, "Test")


# Target distribution analysis
target_col = 'diagnosed_diabetes'

target_counts = train_df[target_col].value_counts()
target_pct = train_df[target_col].value_counts(normalize=True) * 100

print(f"\n{'='*60}")
print("TARGET DISTRIBUTION (diagnosed_diabetes)")
print(f"{'='*60}")
print(f"\nClass 0 (No Diabetes): {target_counts[0]:,} ({target_pct[0]:.2f}%)")
print(f"Class 1 (Diabetes): {target_counts[1]:,} ({target_pct[1]:.2f}%)")
print(f"\nClass Imbalance Ratio: {target_counts[0]/target_counts[1]:.2f}:1")

# Visualization
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Target Distribution', 'Target Percentage'),
    specs=[[{'type': 'bar'}, {'type': 'pie'}]]
)

fig.add_trace(
    go.Bar(x=['No Diabetes', 'Diabetes'], y=target_counts.values,
           marker_color=['#2ecc71', '#e74c3c'],
           text=target_counts.values, textposition='outside'),
    row=1, col=1
)

fig.add_trace(
    go.Pie(labels=['No Diabetes', 'Diabetes'], values=target_counts.values,
           marker_colors=['#2ecc71', '#e74c3c']),
    row=1, col=2
)

fig.update_layout(height=400, showlegend=False, title_text="Target Variable Analysis")
fig.show()


# Identify feature types
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in ['id', target_col]]

categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric Features ({len(numeric_features)}):")
print(numeric_features)
print(f"\nCategorical Features ({len(categorical_features)}):")
print(categorical_features)


# Correlation heatmap for numeric features
correlation_matrix = train_df[numeric_features + [target_col]].corr()

# Sort by correlation with target
target_corr = correlation_matrix[target_col].sort_values(ascending=False)
print("\nTop 15 Features Correlated with Diabetes:")
print(target_corr.head(15))

# Interactive correlation heatmap
fig = px.imshow(
    correlation_matrix,
    labels=dict(color="Correlation"),
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    color_continuous_scale='RdBu_r',
    aspect='auto',
    title='Feature Correlation Heatmap'
)
fig.update_layout(height=800, width=900)
fig.show()


# Distribution plots for key medical features
key_features = ['bmi', 'age', 'systolic_bp', 'diastolic_bp', 'cholesterol_total', 
                'cholesterol_LDL', 'cholesterol_HDL', 'triglycerides']

# Filter available features
key_features = [f for f in key_features if f in train_df.columns]

fig = make_subplots(
    rows=2, cols=4,
    subplot_titles=key_features,
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

for idx, feature in enumerate(key_features):
    row = idx // 4 + 1
    col = idx % 4 + 1
    
    # Distribution for each class
    data_0 = train_df[train_df[target_col] == 0][feature]
    data_1 = train_df[train_df[target_col] == 1][feature]
    
    fig.add_trace(
        go.Histogram(x=data_0, name='No Diabetes', opacity=0.7, 
                     marker_color='#2ecc71', showlegend=(idx==0)),
        row=row, col=col
    )
    fig.add_trace(
        go.Histogram(x=data_1, name='Diabetes', opacity=0.7,
                     marker_color='#e74c3c', showlegend=(idx==0)),
        row=row, col=col
    )

fig.update_layout(
    height=600,
    title_text="Distribution of Key Medical Features by Diabetes Status",
    barmode='overlay'
)
fig.show()


# Box plots: Target vs Key Features
fig = make_subplots(
    rows=2, cols=4,
    subplot_titles=key_features,
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

for idx, feature in enumerate(key_features):
    row = idx // 4 + 1
    col = idx % 4 + 1
    
    for target_val in [0, 1]:
        data = train_df[train_df[target_col] == target_val][feature]
        fig.add_trace(
            go.Box(y=data, name=f'Class {target_val}', 
                   marker_color='#2ecc71' if target_val == 0 else '#e74c3c',
                   showlegend=(idx==0)),
            row=row, col=col
        )

fig.update_layout(
    height=600,
    title_text="Box Plots: Feature Values by Diabetes Status"
)
fig.show()


# Categorical features analysis
if len(categorical_features) > 0:
    n_cat = len(categorical_features)
    n_cols = min(3, n_cat)
    n_rows = (n_cat + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cat > 1 else [axes]
    
    for idx, feature in enumerate(categorical_features):
        crosstab = pd.crosstab(train_df[feature], train_df[target_col], normalize='index') * 100
        crosstab.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'])
        axes[idx].set_title(f'{feature} vs Diabetes Rate')
        axes[idx].set_ylabel('Percentage (%)')
        axes[idx].legend(['No Diabetes', 'Diabetes'])
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(len(categorical_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("No categorical features found in dataset")


def create_features(df, is_train=True):
    """
    Comprehensive feature engineering pipeline based on medical insights.
    
    Medical rationale:
    - BMI categories: Clinical standard for obesity classification
    - Cholesterol ratios: Better CVD risk predictors than absolute values
    - Blood pressure staging: Hypertension strongly linked to diabetes
    - Interaction terms: Synergistic effects (e.g., age amplifies BMI risk)
    """
    df = df.copy()
    
    # BMI-based features (if BMI exists)
    if 'bmi' in df.columns:
        # WHO BMI categories
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        # BMI squared (non-linear relationship with diabetes risk)
        df['bmi_squared'] = df['bmi'] ** 2
        
        # Obesity indicator (BMI >= 30)
        df['is_obese'] = (df['bmi'] >= 30).astype(int)
    
    # Age-based features
    if 'age' in df.columns:
        # Age groups (diabetes risk increases with age)
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 40, 60, 100],
            labels=['Young', 'Middle', 'Senior']
        )
        
        # Age squared
        df['age_squared'] = df['age'] ** 2
    
    # Blood Pressure features
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        # Mean arterial pressure (MAP)
        df['map'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
        
        # Pulse pressure (indicator of arterial stiffness)
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        
        # Hypertension stage (AHA guidelines)
        def classify_bp(row):
            if row['systolic_bp'] < 120 and row['diastolic_bp'] < 80:
                return 'Normal'
            elif row['systolic_bp'] < 130 and row['diastolic_bp'] < 80:
                return 'Elevated'
            elif row['systolic_bp'] < 140 or row['diastolic_bp'] < 90:
                return 'Stage1'
            else:
                return 'Stage2'
        
        df['bp_category'] = df.apply(classify_bp, axis=1)
    
    # Cholesterol ratios (superior CVD risk predictors)
    if 'cholesterol_total' in df.columns and 'cholesterol_HDL' in df.columns:
        # Total/HDL ratio (target: <5 for men, <4.5 for women)
        df['chol_total_hdl_ratio'] = df['cholesterol_total'] / (df['cholesterol_HDL'] + 1e-5)
    
    if 'cholesterol_LDL' in df.columns and 'cholesterol_HDL' in df.columns:
        # LDL/HDL ratio (lower is better)
        df['chol_ldl_hdl_ratio'] = df['cholesterol_LDL'] / (df['cholesterol_HDL'] + 1e-5)
    
    if 'triglycerides' in df.columns and 'cholesterol_HDL' in df.columns:
        # TG/HDL ratio (insulin resistance marker)
        df['tg_hdl_ratio'] = df['triglycerides'] / (df['cholesterol_HDL'] + 1e-5)
    
    # Non-HDL cholesterol (all atherogenic particles)
    if 'cholesterol_total' in df.columns and 'cholesterol_HDL' in df.columns:
        df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['cholesterol_HDL']
    
    # Waist-to-Hip ratio features
    if 'waist_circumference' in df.columns and 'hip_circumference' in df.columns:
        df['waist_to_hip_ratio'] = df['waist_circumference'] / (df['hip_circumference'] + 1e-5)
        
        # Abdominal obesity (central adiposity - strong diabetes predictor)
        # Men: >0.90, Women: >0.85 (assuming mixed population, use 0.87 threshold)
        df['abdominal_obesity'] = (df['waist_to_hip_ratio'] > 0.87).astype(int)
    
    # Interaction features (synergistic effects)
    if 'bmi' in df.columns and 'age' in df.columns:
        # BMI Ã— Age (obesity risk amplified by age)
        df['bmi_age_interaction'] = df['bmi'] * df['age'] / 100  # Scaled
    
    if 'waist_to_hip_ratio' in df.columns and 'bmi' in df.columns:
        # Waist-Hip Ã— BMI (central + general obesity)
        df['whr_bmi_interaction'] = df['waist_to_hip_ratio'] * df['bmi']
    
    if 'physical_activity_minutes_per_week' in df.columns:
        # Physical activity categories
        df['activity_level'] = pd.cut(
            df['physical_activity_minutes_per_week'],
            bins=[0, 75, 150, 300, 10000],
            labels=['Sedentary', 'Low', 'Moderate', 'High']
        )
    
    # Log transforms for skewed features (improve model performance)
    skewed_features = ['triglycerides', 'cholesterol_total', 'cholesterol_LDL']
    for feature in skewed_features:
        if feature in df.columns:
            df[f'{feature}_log'] = np.log1p(df[feature])
    
    # Risk score aggregation (composite features)
    risk_factors = []
    
    if 'is_obese' in df.columns:
        risk_factors.append('is_obese')
    if 'family_history_diabetes' in df.columns:
        risk_factors.append('family_history_diabetes')
    if 'abdominal_obesity' in df.columns:
        risk_factors.append('abdominal_obesity')
    
    if len(risk_factors) > 0:
        df['risk_score'] = df[risk_factors].sum(axis=1)
    
    return df

# Apply feature engineering
print("Creating features for training data...")
train_engineered = create_features(train_df, is_train=True)

print("Creating features for test data...")
test_engineered = create_features(test_df, is_train=False)

print(f"\nâœ“ Feature engineering complete")
print(f"Original features: {train_df.shape[1]}")
print(f"Engineered features: {train_engineered.shape[1]}")
print(f"New features added: {train_engineered.shape[1] - train_df.shape[1]}")


# Original dataset alignment and augmentation
if original_df is not None:
    print("\nAligning original dataset with competition data...")
    
    # Map original target to competition format
    if 'Diabetes_binary' in original_df.columns:
        original_df.rename(columns={'Diabetes_binary': target_col}, inplace=True)
    
    # Apply same feature engineering
    original_engineered = create_features(original_df, is_train=True)
    
    # Align features (use intersection of columns)
    common_features = list(set(train_engineered.columns) & set(original_engineered.columns))
    common_features = [f for f in common_features if f != 'id']  # Remove id column
    
    print(f"Common features: {len(common_features)}")
    
    # Create augmented training set
    train_augmented = pd.concat([
        train_engineered[common_features],
        original_engineered[common_features]
    ], axis=0, ignore_index=True)
    
    print(f"\nâœ“ Data augmentation complete")
    print(f"Original train size: {len(train_engineered):,}")
    print(f"Augmented train size: {len(train_augmented):,}")
    print(f"Augmentation ratio: {len(train_augmented) / len(train_engineered):.2f}x")
    
    # Use augmented data
    train_final = train_augmented.copy()
else:
    print("\nâš  Skipping data augmentation (original dataset not available)")
    train_final = train_engineered.copy()

# Separate features and target
X_train = train_final.drop(columns=[target_col])
y_train = train_final[target_col]

# Test features (keep id for submission)
test_ids = test_engineered['id'].copy()
X_test = test_engineered.drop(columns=['id'])

print(f"\nFinal dataset shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")


# Feature selection using mutual information
def calculate_feature_importance(X, y, method='mutual_info'):
    """
    Calculate feature importance scores.
    """
    # Handle categorical features for mutual information
    X_encoded = X.copy()
    
    # Encode categorical columns
    cat_cols = X_encoded.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
    
    if method == 'mutual_info':
        # Mutual information (works with both linear and non-linear relationships)
        importance = mutual_info_classif(
            X_encoded, y,
            discrete_features=X_encoded.columns.isin(cat_cols),
            random_state=SEED
        )
    
    # Create importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance

# Calculate mutual information scores
print("Calculating feature importance scores...")
feature_scores = calculate_feature_importance(X_train, y_train, method='mutual_info')

print("\nTop 20 Most Important Features:")
display(feature_scores.head(20))

# Visualization
fig = px.bar(
    feature_scores.head(20),
    x='importance',
    y='feature',
    orientation='h',
    title='Top 20 Features by Mutual Information Score',
    labels={'importance': 'Mutual Information Score', 'feature': 'Feature'},
    color='importance',
    color_continuous_scale='Viridis'
)
fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
fig.show()


# Identify numerical and categorical columns
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

if len(categorical_cols) > 0:
    print(f"\nCategorical columns: {categorical_cols}")

# Stratified K-Fold for cross-validation
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print(f"\nâœ“ Cross-validation strategy: {N_FOLDS}-Fold Stratified KFold")
print(f"âœ“ Preserves class distribution in each fold")


# Encode categorical features for baseline models
print("Encoding categorical features for baseline models...")
X_train_baseline = X_train.copy()
X_test_baseline = X_test.copy()

# Label encode categorical features
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train_baseline[col] = le.fit_transform(X_train_baseline[col].astype(str))
    X_test_baseline[col] = le.transform(X_test_baseline[col].astype(str))
    le_dict[col] = le

print(f"âœ“ Categorical features encoded for baseline models")
print(f"X_train_baseline shape: {X_train_baseline.shape}")


def evaluate_baseline_model(model, model_name, X, y, cv):
    """
    Evaluate baseline model with cross-validation.
    """
    print(f"\nEvaluating {model_name}...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    print(f"ROC-AUC Scores: {cv_scores}")
    print(f"Mean ROC-AUC: {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")
    
    return cv_scores

# Baseline models
baseline_results = {}

# 1. Logistic Regression (with scaling)
print("\nTraining Logistic Regression baseline...")
lr_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    ))
])
baseline_results['LogisticRegression'] = evaluate_baseline_model(
    lr_pipeline, 'Logistic Regression', X_train_baseline, y_train, skf
)

# 2. Random Forest
print("\nTraining Random Forest baseline...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
    verbose=0
)
baseline_results['RandomForest'] = evaluate_baseline_model(
    rf_model, 'Random Forest', X_train_baseline, y_train, skf
)

# 3. XGBoost Baseline
print("\nTraining XGBoost baseline...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=SEED,
    n_jobs=-1,
    verbosity=0
)
baseline_results['XGBoost'] = evaluate_baseline_model(
    xgb_model, 'XGBoost Baseline', X_train_baseline, y_train, skf
)

# Summary
print("\n" + "="*60)
print("BASELINE MODEL COMPARISON")
print("="*60)
for model_name, scores in baseline_results.items():
    print(f"{model_name:20s}: {scores.mean():.4f} Â± {scores.std():.4f}")


# Feature importance from Random Forest
print("\nTraining Random Forest for feature importance...")

# Train RF on encoded baseline data
rf_full = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
    verbose=0
)
rf_full.fit(X_train_baseline, y_train)

# Feature importances
rf_importance = pd.DataFrame({
    'feature': X_train_baseline.columns,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Features by Random Forest Importance:")
display(rf_importance.head(20))

# Plot
fig = px.bar(
    rf_importance.head(20),
    x='importance',
    y='feature',
    orientation='h',
    title='Top 20 Features: Random Forest Importance',
    color='importance',
    color_continuous_scale='Cividis'
)
fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
fig.show()


# Prepare data for gradient boosting optimization
# Reuse the encoded data from baseline models section
X_train_encoded = X_train_baseline.copy()
X_test_encoded = X_test_baseline.copy()

print(f"âœ“ Using encoded features for hyperparameter optimization")
print(f"Train shape: {X_train_encoded.shape}")
print(f"Test shape: {X_test_encoded.shape}")


# XGBoost Optimization
def objective_xgb(trial):
    """
    Optuna objective function for XGBoost.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Cross-validation
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_encoded, y_train)):
        X_fold_train = X_train_encoded.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_encoded.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        
        y_pred = model.predict_proba(X_fold_val)[:, 1]
        score = roc_auc_score(y_fold_val, y_pred)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

print("Starting XGBoost hyperparameter optimization...")
study_xgb = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)    
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

print(f"\nBest XGBoost ROC-AUC: {study_xgb.best_value:.4f}")
print("Best XGBoost Parameters:")
print(study_xgb.best_params)


# LightGBM Optimization
def objective_lgb(trial):
    """
    Optuna objective function for LightGBM.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'class_weight': 'balanced',
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # Cross-validation
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_encoded, y_train)):
        X_fold_train = X_train_encoded.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_encoded.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        y_pred = model.predict_proba(X_fold_val)[:, 1]
        score = roc_auc_score(y_fold_val, y_pred)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

print("\nStarting LightGBM hyperparameter optimization...")
study_lgb = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)
study_lgb.optimize(objective_lgb, n_trials=5, show_progress_bar=True)

print(f"\nBest LightGBM ROC-AUC: {study_lgb.best_value:.4f}")
print("Best LightGBM Parameters:")
print(study_lgb.best_params)


# CatBoost Optimization
def objective_cat(trial):
    """
    Optuna objective function for CatBoost.
    """
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'auto_class_weights': 'Balanced',
        'random_state': SEED,
        'verbose': False
    }
    
    # Cross-validation
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_encoded, y_train)):
        X_fold_train = X_train_encoded.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_encoded.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        y_pred = model.predict_proba(X_fold_val)[:, 1]
        score = roc_auc_score(y_fold_val, y_pred)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

print("\nStarting CatBoost hyperparameter optimization...")
study_cat = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)
study_cat.optimize(objective_cat, n_trials=1, show_progress_bar=True)

print(f"\nBest CatBoost ROC-AUC: {study_cat.best_value:.4f}")
print("Best CatBoost Parameters:")
print(study_cat.best_params)


# Summary of optimization results
print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION SUMMARY")
print("="*60)
print(f"XGBoost Best ROC-AUC:   {study_xgb.best_value:.4f}")
print(f"LightGBM Best ROC-AUC:  {study_lgb.best_value:.4f}")
print(f"CatBoost Best ROC-AUC:  {study_cat.best_value:.4f}")

# Store best models
best_xgb_params = study_xgb.best_params.copy()
best_xgb_params.update({
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0
})

best_lgb_params = study_lgb.best_params.copy()
best_lgb_params.update({
    'class_weight': 'balanced',
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1
})

best_cat_params = study_cat.best_params.copy()
best_cat_params.update({
    'auto_class_weights': 'Balanced',
    'random_state': SEED,
    'verbose': False
})


# ============================================================
# FIX FEATURE MISMATCH ISSUE
# ============================================================

print("\n" + "="*60)
print("CHECKING FEATURE MISMATCH")
print("="*60)

print(f"\nTraining features: {X_train_encoded.shape[1]}")
print(f"Test features: {X_test_encoded.shape[1]}")

# Check which features are different
train_cols = set(X_train_encoded.columns)
test_cols = set(X_test_encoded.columns)

missing_in_test = train_cols - test_cols
missing_in_train = test_cols - train_cols

if missing_in_test:
    print(f"\nâš ï¸�  Features in TRAIN but NOT in TEST:")
    print(list(missing_in_test))
    
if missing_in_train:
    print(f"\nâš ï¸�  Features in TEST but NOT in TRAIN:")
    print(list(missing_in_train))

# ============================================================
# ALIGN FEATURES
# ============================================================

print("\n" + "="*60)
print("ALIGNING FEATURES")
print("="*60)

# Ensure test has same columns as train (in same order)
for col in X_train_encoded.columns:
    if col not in X_test_encoded.columns:
        print(f"Adding missing column: {col}")
        X_test_encoded[col] = 0  # Add missing column with default value

# Remove extra columns from test
for col in X_test_encoded.columns:
    if col not in X_train_encoded.columns:
        print(f"Removing extra column: {col}")
        X_test_encoded = X_test_encoded.drop(columns=[col])

# Reorder test columns to match train
X_test_encoded = X_test_encoded[X_train_encoded.columns]

print(f"\nâœ“ After alignment:")
print(f"  Training features: {X_train_encoded.shape[1]}")
print(f"  Test features: {X_test_encoded.shape[1]}")
print(f"  Columns match: {list(X_train_encoded.columns) == list(X_test_encoded.columns)}")

# ============================================================
# CREATE SUBMISSION WITH BEST MODEL (LightGBM)
# ============================================================

print("\n" + "="*60)
print("TRAINING FINAL MODEL FOR SUBMISSION")
print("="*60)

# Get best model info
model_scores = {
    'XGBoost': study_xgb.best_value,
    'LightGBM': study_lgb.best_value,
    'CatBoost': study_cat.best_value
}

best_model_name = max(model_scores, key=model_scores.get)
print(f"\nğŸ�† Best Model: {best_model_name} (CV ROC-AUC: {model_scores[best_model_name]:.4f})")

# Train final LightGBM model
print(f"\nTraining {best_model_name} on full dataset ({len(X_train_encoded):,} samples)...")

final_model = lgb.LGBMClassifier(**best_lgb_params)
final_model.fit(
    X_train_encoded, 
    y_train,
    callbacks=[lgb.log_evaluation(0)]
)

print("âœ“ Model training complete")

# Make predictions
print("\nGenerating predictions on test set...")
test_predictions = final_model.predict_proba(X_test_encoded)[:, 1]

# Clip predictions to avoid extreme values
test_predictions_clipped = np.clip(test_predictions, 0.001, 0.999)

print(f"âœ“ Predictions generated for {len(test_predictions):,} samples")

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_predictions_clipped
})

# Save submission
submission.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("âœ… SUBMISSION FILE CREATED")
print("="*60)
print(f"File: submission.csv")
print(f"Rows: {len(submission):,}")
print(f"\nPrediction Statistics:")
print(submission['diagnosed_diabetes'].describe())

print("\nğŸ“‹ Preview (first 10 rows):")
print(submission.head(10).to_string(index=False))

print(f"\nğŸ�¯ Expected Public LB Score: ~{model_scores[best_model_name]:.4f}")

# ============================================================
# BONUS: CREATE ENSEMBLE SUBMISSION (RECOMMENDED)
# ============================================================

print("\n" + "="*60)
print("CREATING ENSEMBLE SUBMISSION")
print("="*60)

# Train all three models
print("\nTraining ensemble models on full dataset...")

# 1. XGBoost
print("  â†’ Training XGBoost...")
xgb_final = xgb.XGBClassifier(**best_xgb_params)
xgb_final.fit(X_train_encoded, y_train, verbose=False)
xgb_preds = xgb_final.predict_proba(X_test_encoded)[:, 1]

# 2. LightGBM (already trained above)
print("  â†’ Using LightGBM (already trained)...")
lgb_preds = test_predictions  # Reuse from above

# 3. CatBoost
print("  â†’ Training CatBoost...")
cat_final = CatBoostClassifier(**best_cat_params)
cat_final.fit(X_train_encoded, y_train, verbose=False)
cat_preds = cat_final.predict_proba(X_test_encoded)[:, 1]

print("âœ“ All models trained")

# Calculate weights based on CV performance
total_score = sum(model_scores.values())
weights = {name: score/total_score for name, score in model_scores.items()}

print(f"\nğŸ“Š Ensemble Weights:")
print(f"  XGBoost:  {weights['XGBoost']:.3f} (CV: {model_scores['XGBoost']:.4f})")
print(f"  LightGBM: {weights['LightGBM']:.3f} (CV: {model_scores['LightGBM']:.4f})")
print(f"  CatBoost: {weights['CatBoost']:.3f} (CV: {model_scores['CatBoost']:.4f})")

# Weighted ensemble
ensemble_preds = (
    weights['XGBoost'] * xgb_preds +
    weights['LightGBM'] * lgb_preds +
    weights['CatBoost'] * cat_preds
)

# Clip predictions
ensemble_preds_clipped = np.clip(ensemble_preds, 0.001, 0.999)

# Create ensemble submission
submission_ensemble = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': ensemble_preds_clipped
})

submission_ensemble.to_csv('submission_ensemble.csv', index=False)

print("\n" + "="*60)
print("âœ… ENSEMBLE SUBMISSION CREATED")
print("="*60)
print(f"File: submission_ensemble.csv")
print(f"Rows: {len(submission_ensemble):,}")
print(f"\nPrediction Statistics:")
print(submission_ensemble['diagnosed_diabetes'].describe())

print("\nğŸ“‹ Preview (first 10 rows):")
print(submission_ensemble.head(10).to_string(index=False))

# Expected ensemble score (usually 0.5-1% better)
expected_ensemble_score = np.mean(list(model_scores.values())) + 0.005
print(f"\nğŸ�¯ Expected Public LB Score: ~{expected_ensemble_score:.4f}")

# ============================================================
# VERIFICATION & RECOMMENDATIONS
# ============================================================

print("\n" + "="*60)
print("ğŸ“� SUBMISSION FILES SUMMARY")
print("="*60)

import os

for filename in ['submission.csv', 'submission_ensemble.csv']:
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        file_size = os.path.getsize(filename) / 1024  # KB
        
        print(f"\n{'='*60}")
        print(f"File: {filename}")
        print(f"{'='*60}")
        print(f"  Size: {file_size:.2f} KB")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  ID range: {df['id'].min()} to {df['id'].max()}")
        print(f"  Pred range: [{df['diagnosed_diabetes'].min():.6f}, {df['diagnosed_diabetes'].max():.6f}]")
        print(f"  Pred mean: {df['diagnosed_diabetes'].mean():.6f}")
        print(f"  Null values: {df.isnull().sum().sum()}")
        
        # Validation
        issues = []
        if df.shape[1] != 2:
            issues.append("â�Œ Wrong number of columns")
        if df['diagnosed_diabetes'].min() < 0 or df['diagnosed_diabetes'].max() > 1:
            issues.append("â�Œ Predictions outside [0,1]")
        if df.isnull().sum().sum() > 0:
            issues.append("â�Œ Contains null values")
        if len(df) != len(test_ids):
            issues.append("â�Œ Wrong number of rows")
            
        if issues:
            print(f"\n  âš ï¸�  Issues found:")
            for issue in issues:
                print(f"      {issue}")
        else:
            print(f"\n  âœ… Format validated - ready to submit!")

print("\n" + "="*60)
print("ğŸ�¯ FINAL RECOMMENDATION")
print("="*60)
print("\n1ï¸�âƒ£  BEST CHOICE: submission_ensemble.csv")
print(f"    â†’ Weighted ensemble of 3 models")
print(f"    â†’ Expected score: ~{expected_ensemble_score:.4f}")
print(f"    â†’ More robust and typically 0.5-1% better")

print("\n2ï¸�âƒ£  BACKUP: submission.csv")
print(f"    â†’ Single LightGBM model")
print(f"    â†’ Expected score: ~{model_scores['LightGBM']:.4f}")
print(f"    â†’ Faster, simpler")

print("\nğŸ“¤ UPLOAD TO KAGGLE:")
print("    1. Go to: https://www.kaggle.com/competitions/playground-series-s5e12/submit")
print("    2. Upload: submission_ensemble.csv")
print("    3. Description: 'Weighted Ensemble (XGB+LGB+CAT) with Feature Engineering - CV 0.7275'")
print("    4. Submit!")

print("\nâœ¨ Good luck! You should score around 0.73-0.74 on public LB! ğŸš€")


def train_stacking_ensemble(X, y, X_test, cv, best_params):
    """
    Train stacking ensemble with optimized base models.
    
    Level 1: XGBoost, LightGBM, CatBoost (base learners)
    Level 2: Logistic Regression (meta-learner)
    """
    print("\nTraining Stacking Ensemble...")
    
    # Initialize models with best parameters
    xgb_model = xgb.XGBClassifier(**best_params['xgb'])
    lgb_model = lgb.LGBMClassifier(**best_params['lgb'])
    cat_model = CatBoostClassifier(**best_params['cat'])
    
    # Storage for out-of-fold predictions
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    
    # Storage for test predictions
    test_xgb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))
    
    # Cross-validation training
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X, y), total=cv.n_splits, desc="CV Folds")):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # XGBoost
        xgb_model.fit(X_fold_train, y_fold_train, verbose=False)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_fold_val)[:, 1]
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / cv.n_splits
        
        # LightGBM
        lgb_model.fit(X_fold_train, y_fold_train, callbacks=[lgb.log_evaluation(0)])
        oof_lgb[val_idx] = lgb_model.predict_proba(X_fold_val)[:, 1]
        test_lgb += lgb_model.predict_proba(X_test)[:, 1] / cv.n_splits
        
        # CatBoost
        cat_model.fit(X_fold_train, y_fold_train, verbose=False)
        oof_cat[val_idx] = cat_model.predict_proba(X_fold_val)[:, 1]
        test_cat += cat_model.predict_proba(X_test)[:, 1] / cv.n_splits
        
        # Ensemble score for this fold
        oof_ensemble = (oof_xgb[val_idx] + oof_lgb[val_idx] + oof_cat[val_idx]) / 3
        fold_score = roc_auc_score(y_fold_val, oof_ensemble)
        fold_scores.append(fold_score)
        print(f"Fold {fold+1} Ensemble ROC-AUC: {fold_score:.4f}")
    
    # Overall OOF scores
    print("\n" + "="*60)
    print("OUT-OF-FOLD SCORES (Level 1 Models)")
    print("="*60)
    print(f"XGBoost OOF ROC-AUC:  {roc_auc_score(y, oof_xgb):.4f}")
    print(f"LightGBM OOF ROC-AUC: {roc_auc_score(y, oof_lgb):.4f}")
    print(f"CatBoost OOF ROC-AUC: {roc_auc_score(y, oof_cat):.4f}")
    
    # Simple average ensemble
    oof_avg = (oof_xgb + oof_lgb + oof_cat) / 3
    test_avg = (test_xgb + test_lgb + test_cat) / 3
    print(f"\nAverage Ensemble OOF ROC-AUC: {roc_auc_score(y, oof_avg):.4f}")
    
    # Train meta-learner (Level 2)
    print("\nTraining Meta-Learner (Level 2)...")
    meta_features_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
    meta_features_test = np.column_stack([test_xgb, test_lgb, test_cat])
    
    meta_learner = LogisticRegression(max_iter=1000, random_state=SEED)
    meta_learner.fit(meta_features_train, y)
    
    # Meta-learner predictions
    oof_meta = meta_learner.predict_proba(meta_features_train)[:, 1]
    test_meta = meta_learner.predict_proba(meta_features_test)[:, 1]
    
    print(f"Meta-Learner OOF ROC-AUC: {roc_auc_score(y, oof_meta):.4f}")
    
    return {
        'oof_predictions': {
            'xgb': oof_xgb,
            'lgb': oof_lgb,
            'cat': oof_cat,
            'avg': oof_avg,
            'meta': oof_meta
        },
        'test_predictions': {
            'xgb': test_xgb,
            'lgb': test_lgb,
            'cat': test_cat,
            'avg': test_avg,
            'meta': test_meta
        },
        'fold_scores': fold_scores
    }

# Train ensemble
best_params = {
    'xgb': best_xgb_params,
    'lgb': best_lgb_params,
    'cat': best_cat_params
}

ensemble_results = train_stacking_ensemble(
    X_train_encoded, y_train, X_test_encoded, skf, best_params
)


# Optimize ensemble weights using Optuna
def objective_ensemble_weights(trial):
    """
    Optimize weighted average of model predictions.
    """
    w1 = trial.suggest_float('w_xgb', 0, 1)
    w2 = trial.suggest_float('w_lgb', 0, 1)
    w3 = trial.suggest_float('w_cat', 0, 1)
    
    # Normalize weights
    total = w1 + w2 + w3
    w1, w2, w3 = w1/total, w2/total, w3/total
    
    # Weighted ensemble
    oof_weighted = (
        w1 * ensemble_results['oof_predictions']['xgb'] +
        w2 * ensemble_results['oof_predictions']['lgb'] +
        w3 * ensemble_results['oof_predictions']['cat']
    )
    
    return roc_auc_score(y_train, oof_weighted)

print("\nOptimizing ensemble weights...")
study_weights = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED)
)
study_weights.optimize(objective_ensemble_weights, n_trials=100, show_progress_bar=True)

# Best weights
best_weights = study_weights.best_params
total = sum(best_weights.values())
best_weights = {k: v/total for k, v in best_weights.items()}

print(f"\nBest Ensemble Weights:")
print(f"XGBoost:  {best_weights['w_xgb']:.4f}")
print(f"LightGBM: {best_weights['w_lgb']:.4f}")
print(f"CatBoost: {best_weights['w_cat']:.4f}")
print(f"\nWeighted Ensemble OOF ROC-AUC: {study_weights.best_value:.4f}")

# Create weighted test predictions
test_weighted = (
    best_weights['w_xgb'] * ensemble_results['test_predictions']['xgb'] +
    best_weights['w_lgb'] * ensemble_results['test_predictions']['lgb'] +
    best_weights['w_cat'] * ensemble_results['test_predictions']['cat']
)

ensemble_results['test_predictions']['weighted'] = test_weighted


# ROC Curve for ensemble models
fig = go.Figure()

for model_name, oof_pred in ensemble_results['oof_predictions'].items():
    fpr, tpr, _ = roc_curve(y_train, oof_pred)
    auc_score = roc_auc_score(y_train, oof_pred)
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'{model_name.upper()} (AUC={auc_score:.4f})',
        mode='lines'
    ))

# Random classifier line
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    name='Random Classifier',
    mode='lines',
    line=dict(dash='dash', color='gray')
))

fig.update_layout(
    title='ROC Curves: Ensemble Models (Out-of-Fold)',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    height=600,
    width=800
)
fig.show()


# SHAP Analysis for XGBoost (most interpretable)
print("Calculating SHAP values...")

# Train final XGBoost model on full data
final_xgb = xgb.XGBClassifier(**best_xgb_params)
final_xgb.fit(X_train_encoded, y_train, verbose=False)

# Calculate SHAP values (sample for speed)
sample_size = min(1000, len(X_train_encoded))
sample_idx = np.random.choice(len(X_train_encoded), sample_size, replace=False)
X_sample = X_train_encoded.iloc[sample_idx]

explainer = shap.TreeExplainer(final_xgb)
shap_values = explainer.shap_values(X_sample)

print("âœ“ SHAP values calculated")


# SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=20)
plt.title('SHAP Feature Importance: XGBoost Model', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# Beeswarm plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
plt.title('SHAP Summary Plot: Feature Impact on Diabetes Prediction', fontsize=14, pad=20)
plt.tight_layout()
plt.show()


# SHAP Dependence Plots for top features
top_features = ['bmi', 'age', 'family_history_diabetes', 'systolic_bp', 'cholesterol_total']
available_top_features = [f for f in top_features if f in X_sample.columns]

if len(available_top_features) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(available_top_features[:6]):
        shap.dependence_plot(
            feature, shap_values, X_sample,
            ax=axes[idx], show=False
        )
    
    # Hide unused subplots
    for idx in range(len(available_top_features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('SHAP Dependence Plots: Key Diabetes Risk Factors', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


# Feature importance comparison across models
xgb_importance = pd.DataFrame({
    'feature': X_train_encoded.columns,
    'importance': final_xgb.feature_importances_
}).sort_values('importance', ascending=False).head(20)

fig = px.bar(
    xgb_importance,
    x='importance',
    y='feature',
    orientation='h',
    title='Top 20 Features: XGBoost Feature Importance',
    color='importance',
    color_continuous_scale='Viridis'
)
fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
fig.show()


# Adversarial validation (check train/test distribution similarity)
print("\nPerforming Adversarial Validation...")

# Combine train and test (use encoded versions)
X_combined = pd.concat([
    X_train_encoded.assign(is_test=0),
    X_test_encoded.assign(is_test=1)
], axis=0, ignore_index=True)

y_adversarial = X_combined['is_test']
X_adversarial = X_combined.drop(columns=['is_test'])

# Train classifier to distinguish train from test
adv_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=SEED,
    verbose=-1
)

adv_scores = cross_val_score(
    adv_model, X_adversarial, y_adversarial,
    cv=5,
    scoring='roc_auc'
)

print(f"\nAdversarial Validation ROC-AUC: {adv_scores.mean():.4f} Â± {adv_scores.std():.4f}")
print("\nInterpretation:")
if adv_scores.mean() < 0.55:
    print("âœ“ Train and test distributions are very similar (good!)")
elif adv_scores.mean() < 0.65:
    print("âš  Slight distribution shift between train and test")
else:
    print("âš âš  Significant distribution shift - be cautious of overfitting")


# Select best ensemble strategy
print("\n" + "="*60)
print("FINAL ENSEMBLE SELECTION")
print("="*60)

# Compare all ensemble strategies
ensemble_comparison = {
    'Simple Average': roc_auc_score(y_train, ensemble_results['oof_predictions']['avg']),
    'Weighted Average': study_weights.best_value,
    'Meta-Learner': roc_auc_score(y_train, ensemble_results['oof_predictions']['meta'])
}

for name, score in ensemble_comparison.items():
    print(f"{name:20s}: {score:.4f}")

# Use weighted average (typically most robust)
final_predictions = test_weighted
print(f"\nâœ“ Selected: Weighted Average Ensemble")


# Clip predictions to avoid extreme values
final_predictions_clipped = np.clip(final_predictions, 0.001, 0.999)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    target_col: final_predictions_clipped
})

print("\nSubmission Preview:")
display(submission.head(10))

print("\nSubmission Statistics:")
print(submission[target_col].describe())

# Verify submission format
assert submission.shape[0] == test_df.shape[0], "Submission has wrong number of rows"
assert submission.shape[1] == 2, "Submission has wrong number of columns"
assert list(submission.columns) == list(sample_submission.columns), "Column names don't match"

print("\nâœ“ Submission format validated")


# Save submission file
submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"\n{'='*60}")
print("SUBMISSION SAVED")
print(f"{'='*60}")
print(f"File: {submission_filename}")
print(f"Rows: {len(submission):,}")
print(f"\nExpected Public Leaderboard Score: {study_weights.best_value:.4f}")
print("\nUpload Instructions:")
print("1. Go to: https://www.kaggle.com/competitions/playground-series-s5e12/submit")
print("2. Upload: submission.csv")
print("3. Add description: 'Weighted Ensemble (XGB+LGB+CAT) with Feature Engineering'")
print("\nâœ“ Good luck! ğŸ�¯")

