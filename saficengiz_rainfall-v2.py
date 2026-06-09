# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Load datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

print('Train columns:', df_train.columns.tolist())
print('Test columns:', df_test.columns.tolist())

print('Train shape:', df_train.shape)
print('Test shape:', df_test.shape)


# Initial data exploration and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Check for class imbalance in the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='rainfall', data=df_train)
plt.title('Class Distribution in Train Set')
plt.show()


# Correlation heatmap for feature exploration
corr = df_train.corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, 
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
)
plt.title('Correlation Heatmap of Features', fontsize=16, pad=20)
plt.tight_layout()
plt.show()


# Create a combined dataframe for preprocessing
df_nlarge = pd.concat([df_train, df_test], axis=0)


# Save test IDs for later submission
df_test_id = df_nlarge.loc[~df_nlarge["rainfall"].notna(), "id"].copy()


# Drop ID column as it's not useful for modeling
df_nlarge.drop(["id"], axis=1, inplace=True)


# Check for missing values
print("Missing values in combined dataset:")
print(df_nlarge.isnull().sum())


# Fill missing values
df_nlarge['winddirection'].fillna(df_nlarge['winddirection'].median(), inplace=True)

# Double-check that all missing values are handled
print("\nRemaining missing values after imputation:")
print(df_nlarge.isnull().sum())


# Feature Engineering
print("\nPerforming feature engineering...")

# Z-score normalization (standard scaling)
df_nlarge['temp_normalized'] = (df_nlarge['temparature'] - df_nlarge['temparature'].mean()) / df_nlarge['temparature'].std()

# Create interaction features
df_nlarge['humidity_dewpoint_interaction'] = df_nlarge['humidity'] * df_nlarge['dewpoint']
df_nlarge['humidity_cloud'] = df_nlarge['humidity'] * df_nlarge['cloud'] / 100

# Create temporal features
df_nlarge['sin_day'] = np.sin(2 * np.pi * df_nlarge['day']/365)  # Seasonal component
df_nlarge['cos_day'] = np.cos(2 * np.pi * df_nlarge['day']/365)  # Seasonal component

# Create weather-specific features
df_nlarge['dewpoint_diff'] = df_nlarge['temparature'] - df_nlarge['dewpoint']
df_nlarge['sunshine_cloud_ratio'] = df_nlarge['sunshine'] / (df_nlarge['cloud'] + 1)  # Adding 1 to avoid division by zero
df_nlarge['pressure_normalized'] = (df_nlarge['pressure'] - df_nlarge['pressure'].mean()) / df_nlarge['pressure'].std()
df_nlarge['humidity_temparature_ratio'] = df_nlarge['humidity'] / (df_nlarge['temparature'] + 1)  # Adding 1 to avoid division by zero

# Wind features
df_nlarge['wind_direction_rad'] = np.radians(df_nlarge['winddirection'])
df_nlarge['wind_x'] = df_nlarge['windspeed'] * np.cos(df_nlarge['wind_direction_rad'])
df_nlarge['wind_y'] = df_nlarge['windspeed'] * np.sin(df_nlarge['wind_direction_rad'])
df_nlarge['wind_chill'] = 13.12 + 0.6215*df_nlarge['temparature'] - 11.37*(df_nlarge['windspeed']**0.16) + 0.3965*df_nlarge['temparature']*(df_nlarge['windspeed']**0.16)

#Bonuses
df_nlarge["dew_humidity/sun"] = df_nlarge["dewpoint"] * df_nlarge["humidity"] / (df_nlarge['sunshine'] + 1)
df_nlarge['cloud_sun_ratio'] = df_nlarge['cloud'] / (df_nlarge['sunshine'] + 1)
df_nlarge['humidity_sunshine_*'] = df_nlarge["humidity"] * df_nlarge['sunshine']


# Polynomial and interaction features for high correlation variables
df_nlarge['cloud_squared'] = df_nlarge['cloud'] ** 2
df_nlarge['humidity_squared'] = df_nlarge['humidity'] ** 2
df_nlarge['sunshine_squared'] = df_nlarge['sunshine'] ** 2
df_nlarge['cloud_humidity_interaction'] = df_nlarge['cloud'] * df_nlarge['humidity']
df_nlarge['sunshine_humidity_interaction'] = df_nlarge['sunshine'] * df_nlarge['humidity']


# Drop less useful columns
df_nlarge.drop(["mintemp", "maxtemp", "day", "wind_direction_rad"], axis=1, inplace=True)

# Final check for missing values
print("\nFinal check for missing values:")
print(df_nlarge.isnull().sum())


# Split back into train and test
df_train_ = df_nlarge.loc[df_nlarge["rainfall"].notna()]
df_test_ = df_nlarge.loc[~df_nlarge["rainfall"].notna()]


# Define target and features
y = df_train_["rainfall"]
X = df_train_.drop("rainfall", axis=1)


# Train-test split
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print class distribution
print("\nClass distribution in training set:")
print(y_train.value_counts())
print("\nClass distribution in test set:")
print(y_test.value_counts())


from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
from sklearn.metrics import roc_auc_score
import optuna
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


# Ensure data has no missing values
print("Checking for missing values in training data...")
print("Missing values in X_train:", X_train.isnull().sum().sum())
print("Missing values in X_test:", X_test.isnull().sum().sum())


# If any missing values exist, impute them
if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    df_test_ = pd.DataFrame(imputer.transform(df_test_), columns=df_test_.columns)
    print("After imputation - Missing values in X_train:", X_train.isnull().sum().sum())
    print("After imputation - Missing values in X_test:", X_test.isnull().sum().sum())



# Check class distribution
print("\nOriginal class distribution:")
print(Counter(y_train))


# Define a simple function to evaluate models with different resampling techniques
def evaluate_resampling(X_train, y_train, X_test, y_test, resampler=None, name="No resampling"):
    if resampler is not None:
        X_res, y_res = resampler.fit_resample(X_train, y_train)
        print(f"{name} - New class distribution: {Counter(y_res)}")
    else:
        X_res, y_res = X_train, y_train
        print(f"{name} - Original class distribution: {Counter(y_res)}")
    
    # Use a simple model for quick evaluation
    model = LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
    model.fit(X_res, y_res)
    y_pred = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred)
    print(f"{name} - ROC AUC: {score:.4f}")
    return score, X_res, y_res

# Test different resampling techniques
results = {}

# No resampling (baseline)
baseline_score, _, _ = evaluate_resampling(X_train, y_train, X_test, y_test, name="Baseline")
results["Baseline"] = baseline_score

# SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Adjust sampling strategy for mild oversampling
smote_score, X_smote, y_smote = evaluate_resampling(X_train, y_train, X_test, y_test, smote, "SMOTE")
results["SMOTE"] = smote_score

# BorderlineSMOTE
bsmote = BorderlineSMOTE(random_state=42, sampling_strategy=0.5)
bsmote_score, X_bsmote, y_bsmote = evaluate_resampling(X_train, y_train, X_test, y_test, bsmote, "BorderlineSMOTE")
results["BorderlineSMOTE"] = bsmote_score

# ADASYN
adasyn = ADASYN(random_state=42, sampling_strategy=0.5)
adasyn_score, X_adasyn, y_adasyn = evaluate_resampling(X_train, y_train, X_test, y_test, adasyn, "ADASYN")
results["ADASYN"] = adasyn_score

# Random Under Sampling
rus = RandomUnderSampler(random_state=42, sampling_strategy=0.8)
rus_score, X_rus, y_rus = evaluate_resampling(X_train, y_train, X_test, y_test, rus, "Random Undersampling")
results["Random Undersampling"] = rus_score

# SMOTETomek (combination)
smote_tomek = SMOTETomek(random_state=42, sampling_strategy=0.5)
smote_tomek_score, X_smote_tomek, y_smote_tomek = evaluate_resampling(X_train, y_train, X_test, y_test, smote_tomek, "SMOTE+Tomek")
results["SMOTE+Tomek"] = smote_tomek_score

# Select the best resampling technique
best_technique = max(results, key=results.get)
print(f"\nBest resampling technique: {best_technique} with AUC: {results[best_technique]:.4f}")


# Use the best resampling technique for final model training
if best_technique == "SMOTE":
    X_resampled, y_resampled = X_smote, y_smote
elif best_technique == "BorderlineSMOTE":
    X_resampled, y_resampled = X_bsmote, y_bsmote
elif best_technique == "ADASYN":
    X_resampled, y_resampled = X_adasyn, y_adasyn
elif best_technique == "Random Undersampling":
    X_resampled, y_resampled = X_rus, y_rus
elif best_technique == "SMOTE+Tomek":
    X_resampled, y_resampled = X_smote_tomek, y_smote_tomek
else:
    X_resampled, y_resampled = X_train, y_train  # Use original data if baseline is best

print("\nFinal resampled training data shape:", X_resampled.shape)
print("Final resampled class distribution:", Counter(y_resampled))


#------------------------------------------------------------------------------
# OPTIMIZE MODELS WITH RESAMPLED DATA
#------------------------------------------------------------------------------
print("\nOptimizing models with resampled data...")

# Set Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Define LightGBM optimization function
def objective_lgbm(trial):
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'verbose': -1,
        'random_state': 42
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, valid_idx in cv.split(X_resampled, y_resampled):
        X_train_fold, X_valid_fold = X_resampled.iloc[train_idx], X_resampled.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_resampled.iloc[train_idx], y_resampled.iloc[valid_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict_proba(X_valid_fold)[:, 1]
        score = roc_auc_score(y_valid_fold, y_pred)
        scores.append(score)
    
    return np.mean(scores)


# Define XGBoost optimization function
def objective_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'random_state': 42,
        'verbosity': 0
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, valid_idx in cv.split(X_resampled, y_resampled):
        X_train_fold, X_valid_fold = X_resampled.iloc[train_idx], X_resampled.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_resampled.iloc[train_idx], y_resampled.iloc[valid_idx]
        
        model = XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict_proba(X_valid_fold)[:, 1]
        score = roc_auc_score(y_valid_fold, y_pred)
        scores.append(score)
    
    return np.mean(scores)


# Define CatBoost optimization function
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
        'random_seed': 42,
        'verbose': False
    }
    
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, valid_idx in cv.split(X_resampled, y_resampled):
        X_train_fold, X_valid_fold = X_resampled.iloc[train_idx], X_resampled.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_resampled.iloc[train_idx], y_resampled.iloc[valid_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_train_fold, y_train_fold, verbose=False)
        y_pred = model.predict_proba(X_valid_fold)[:, 1]
        score = roc_auc_score(y_valid_fold, y_pred)
        scores.append(score)
    
    return np.mean(scores)


# Run optimization with fewer trials to save time
print("Optimizing LightGBM...")
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=30)  # Reduced from 30 to 10 for time
print(f"Best LightGBM AUC: {study_lgbm.best_value:.4f}")
print(f"Best LightGBM params: {study_lgbm.best_params}")

print("\nOptimizing XGBoost...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30)  # Reduced from 30 to 10 for time
print(f"Best XGBoost AUC: {study_xgb.best_value:.4f}")
print(f"Best XGBoost params: {study_xgb.best_params}")

print("\nOptimizing CatBoost...")
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=30)  # Reduced from 30 to 10 for time
print(f"Best CatBoost AUC: {study_cat.best_value:.4f}")
print(f"Best CatBoost params: {study_cat.best_params}")


# Train final models with best parameters
best_lgbm = LGBMClassifier(**study_lgbm.best_params, random_state=42)
best_xgb = XGBClassifier(**study_xgb.best_params, random_state=42)
best_cat = CatBoostClassifier(**study_cat.best_params, random_state=42)

# Train models
print("\nTraining final models...")
best_lgbm.fit(X_resampled, y_resampled)
best_xgb.fit(X_resampled, y_resampled)
best_cat.fit(X_resampled, y_resampled, verbose=False)


# Evaluate models on test set
y_pred_lgbm = best_lgbm.predict_proba(X_test)[:, 1]
y_pred_xgb = best_xgb.predict_proba(X_test)[:, 1]
y_pred_cat = best_cat.predict_proba(X_test)[:, 1]

print("\nTest set performance:")
print(f"LightGBM AUC: {roc_auc_score(y_test, y_pred_lgbm):.4f}")
print(f"XGBoost AUC: {roc_auc_score(y_test, y_pred_xgb):.4f}")
print(f"CatBoost AUC: {roc_auc_score(y_test, y_pred_cat):.4f}")


# Define Logistic Regression optimization function
def objective_lr(trial):
    params = {
        'C': trial.suggest_float('C', 0.001, 10.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
        'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'max_iter': 1000,
        'random_state': 42
    }
    
    # Ensure compatible penalty/solver combinations
    if params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
        params['solver'] = 'saga'
    if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
        params['solver'] = 'liblinear'
    if params['penalty'] is None and params['solver'] in ['liblinear']:
        params['solver'] = 'lbfgs'
    
    # Add L1 ratio only for elasticnet
    if params['penalty'] == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, valid_idx in cv.split(X_resampled, y_resampled):
        X_train_fold, X_valid_fold = X_resampled.iloc[train_idx], X_resampled.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_resampled.iloc[train_idx], y_resampled.iloc[valid_idx]
        
        model = LogisticRegression(**params)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict_proba(X_valid_fold)[:, 1]
        score = roc_auc_score(y_valid_fold, y_pred)
        scores.append(score)
    
    return np.mean(scores)


# Run optimization for Logistic Regression
study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_lr, n_trials=50)  # Same number of trials as other models
print(f"Best Logistic Regression AUC: {study_lr.best_value:.4f}")
print(f"Best Logistic Regression params: {study_lr.best_params}")


# Train final Logistic Regression model with best parameters
best_lr = LogisticRegression(**study_lr.best_params, random_state=42)
best_lr.fit(X_resampled, y_resampled)


# Evaluate on test set
y_pred_lr = best_lr.predict_proba(X_test)[:, 1]
print(f"Logistic Regression AUC: {roc_auc_score(y_test, y_pred_lr):.4f}")


# Update the voting ensemble to include Logistic Regression
voting_clf_with_lr = VotingClassifier(
    estimators=[
        ('lgbm', best_lgbm),
        ('xgb', best_xgb),
        ('cat', best_cat),
        ('lr', best_lr)
    ],
    voting='soft'
)
voting_clf_with_lr.fit(X_resampled, y_resampled)
y_pred_voting_with_lr = voting_clf_with_lr.predict_proba(X_test)[:, 1]
print(f"Voting Ensemble with LR AUC: {roc_auc_score(y_test, y_pred_voting_with_lr):.4f}")


# Update the stacking ensemble to include Logistic Regression
stacking_clf_with_lr = StackingClassifier(
    estimators=[
        ('lgbm', best_lgbm),
        ('xgb', best_xgb),
        ('cat', best_cat),
        ('lr', best_lr)
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced'),
    cv=5,
    stack_method='predict_proba'
)
stacking_clf_with_lr.fit(X_resampled, y_resampled)
y_pred_stacking_with_lr = stacking_clf_with_lr.predict_proba(X_test)[:, 1]
print(f"Stacking Ensemble with LR AUC: {roc_auc_score(y_test, y_pred_stacking_with_lr):.4f}")


# Optimize weights for the blend including Logistic Regression
def optimize_weights_with_lr(preds_list, y_true):
    def objective(trial):
        w1 = trial.suggest_float('w1', 0, 1)
        w2 = trial.suggest_float('w2', 0, 1)
        w3 = trial.suggest_float('w3', 0, 1)
        w4 = trial.suggest_float('w4', 0, 1)
        
        # Normalize weights
        total = w1 + w2 + w3 + w4
        w1, w2, w3, w4 = w1/total, w2/total, w3/total, w4/total
        
        # Weighted blend
        y_blend = w1*preds_list[0] + w2*preds_list[1] + w3*preds_list[2] + w4*preds_list[3]
        return roc_auc_score(y_true, y_blend)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    # Get optimal weights
    w1 = study.best_params['w1']
    w2 = study.best_params['w2']
    w3 = study.best_params['w3']
    w4 = study.best_params['w4']
    total = w1 + w2 + w3 + w4
    return [w1/total, w2/total, w3/total, w4/total], study.best_value


# Optimize weights with LR
print("\nOptimizing blend weights with Logistic Regression...")
weights_with_lr, blend_score_with_lr = optimize_weights_with_lr(
    [y_pred_lgbm, y_pred_xgb, y_pred_cat, y_pred_lr], 
    y_test
)
print(f"Optimal weights: LightGBM={weights_with_lr[0]:.3f}, XGBoost={weights_with_lr[1]:.3f}, CatBoost={weights_with_lr[2]:.3f}, LogisticRegression={weights_with_lr[3]:.3f}")
print(f"Weighted Blend with LR AUC: {blend_score_with_lr:.4f}")


# Create weighted blend predictions with LR
y_pred_blend_with_lr = (weights_with_lr[0]*y_pred_lgbm + 
                        weights_with_lr[1]*y_pred_xgb + 
                        weights_with_lr[2]*y_pred_cat + 
                        weights_with_lr[3]*y_pred_lr)


# Compare all models including LR
model_performance_with_lr = {
    'LightGBM': roc_auc_score(y_test, y_pred_lgbm),
    'XGBoost': roc_auc_score(y_test, y_pred_xgb),
    'CatBoost': roc_auc_score(y_test, y_pred_cat),
    'LogisticRegression': roc_auc_score(y_test, y_pred_lr),
    'Voting': roc_auc_score(y_test, y_pred_voting),
    'Voting with LR': roc_auc_score(y_test, y_pred_voting_with_lr),
    'Stacking': roc_auc_score(y_test, y_pred_stacking),
    'Stacking with LR': roc_auc_score(y_test, y_pred_stacking_with_lr),
    'Weighted Blend': roc_auc_score(y_test, y_pred_blend),
    'Weighted Blend with LR': roc_auc_score(y_test, y_pred_blend_with_lr)
}
best_model_with_lr = max(model_performance_with_lr, key=model_performance_with_lr.get)
print(f"\nBest model: {best_model_with_lr} with AUC: {model_performance_with_lr[best_model_with_lr]:.4f}")


# Generate predictions for submission
lr_preds = best_lr.predict_proba(df_test_)[:, 1]
voting_with_lr_preds = voting_clf_with_lr.predict_proba(df_test_)[:, 1]
stacking_with_lr_preds = stacking_clf_with_lr.predict_proba(df_test_)[:, 1]
blend_with_lr_preds = (weights_with_lr[0]*lgbm_preds + 
                       weights_with_lr[1]*xgb_preds + 
                       weights_with_lr[2]*cat_preds + 
                       weights_with_lr[3]*lr_preds)


# Add to submissions dictionary
submissions['lr'] = pd.DataFrame({'id': df_test_id, 'target': lr_preds})
submissions['voting_with_lr'] = pd.DataFrame({'id': df_test_id, 'target': voting_with_lr_preds})
submissions['stacking_with_lr'] = pd.DataFrame({'id': df_test_id, 'target': stacking_with_lr_preds})
submissions['blend_with_lr'] = pd.DataFrame({'id': df_test_id, 'target': blend_with_lr_preds})

# Save additional submission files
for name in ['lr', 'voting_with_lr', 'stacking_with_lr', 'blend_with_lr']:
    filename = f'submission_{name}.csv'
    submissions[name].to_csv(filename, index=False)
    print(f"Saved {filename}")


# Create a final submission based on the best model including LR
if best_model_with_lr == 'LogisticRegression':
    best_preds = lr_preds
elif best_model_with_lr == 'Voting with LR':
    best_preds = voting_with_lr_preds
elif best_model_with_lr == 'Stacking with LR':
    best_preds = stacking_with_lr_preds
elif best_model_with_lr == 'Weighted Blend with LR':
    best_preds = blend_with_lr_preds
else:
    # Use original best model if it's still better
    if best_model == 'LightGBM':
        best_preds = lgbm_preds
    elif best_model == 'XGBoost':
        best_preds = xgb_preds
    elif best_model == 'CatBoost':
        best_preds = cat_preds
    elif best_model == 'Voting':
        best_preds = voting_preds
    elif best_model == 'Stacking':
        best_preds = stacking_preds
    else:  # Weighted Blend
        best_preds = blend_preds


final_submission_with_lr = pd.DataFrame({'id': df_test_id, 'target': best_preds})
final_submission_with_lr.to_csv('submission_final_with_lr.csv', index=False)
print("\nSaved final submission as submission_final_with_lr.csv")

print("\nAll done! Check the submission files and submit the best one to the competition.")

