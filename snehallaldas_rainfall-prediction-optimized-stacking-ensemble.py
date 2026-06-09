import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import shap
import optuna
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Exploratory Data Analysis
print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nData Types:")
print(train_df.dtypes)
print("\nMissing values in train:")
print(train_df.isnull().sum())
print("\nMissing values in test:")
print(test_df.isnull().sum())

# Plot rainfall distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='rainfall', data=train_df)
plt.title('Rainfall Distribution')
plt.show()

# Correlation analysis
plt.figure(figsize=(14, 10))
correlation = train_df.drop('id', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Fill missing values with appropriate strategies
# For winddirection, we'll fill with median and create a missing indicator
test_df['winddirection_missing'] = test_df['winddirection'].isnull().astype(int)
train_df['winddirection_missing'] = train_df['winddirection'].isnull().astype(int)
test_df["winddirection"].fillna(test_df["winddirection"].median(), inplace=True)
train_df["winddirection"].fillna(train_df["winddirection"].median(), inplace=True)

# Feature Engineering
# Basic features
train_df["temp_diff"] = train_df["maxtemp"] - train_df["mintemp"]
test_df["temp_diff"] = test_df["maxtemp"] - test_df["mintemp"]

train_df["temp_avg"] = (train_df["maxtemp"] + train_df["mintemp"]) / 2
test_df["temp_avg"] = (test_df["maxtemp"] + test_df["mintemp"]) / 2

# Humidity ratio (a measure of how close the air is to saturation)
train_df["humidity_ratio"] = train_df["humidity"] / train_df["temp_avg"]
test_df["humidity_ratio"] = test_df["humidity"] / test_df["temp_avg"]

# Wind chill factor (approximate formula)
train_df["wind_chill"] = 13.12 + 0.6215 * train_df["temp_avg"] - 11.37 * (train_df["windspeed"] ** 0.16) + 0.3965 * train_df["temp_avg"] * (train_df["windspeed"] ** 0.16)
test_df["wind_chill"] = 13.12 + 0.6215 * test_df["temp_avg"] - 11.37 * (test_df["windspeed"] ** 0.16) + 0.3965 * test_df["temp_avg"] * (test_df["windspeed"] ** 0.16)

# Heat index (simplified version)
train_df["heat_index"] = -8.78469475556 + 1.61139411 * train_df["temp_avg"] + 2.33854883889 * train_df["humidity"] - 0.14611605 * train_df["temp_avg"] * train_df["humidity"]
test_df["heat_index"] = -8.78469475556 + 1.61139411 * test_df["temp_avg"] + 2.33854883889 * test_df["humidity"] - 0.14611605 * test_df["temp_avg"] * test_df["humidity"]

# Dew point (approximation)
train_df["dewpoint"] = train_df["temp_avg"] - ((100 - train_df["humidity"]) / 5)
test_df["dewpoint"] = test_df["temp_avg"] - ((100 - test_df["humidity"]) / 5)

# Cyclical encoding for day and winddirection
def encode_cyclical_feature(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)

encode_cyclical_feature(train_df, "day", 365)
encode_cyclical_feature(test_df, "day", 365)
encode_cyclical_feature(train_df, "winddirection", 360)
encode_cyclical_feature(test_df, "winddirection", 360)

# Interaction terms between key meteorological variables
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
weather_vars = ["pressure", "humidity", "windspeed", "temp_avg", "temp_diff"]
interaction_terms = poly.fit_transform(train_df[weather_vars])
interaction_terms_test = poly.transform(test_df[weather_vars])

# Add polynomial features to dataframes
train_poly_df = pd.DataFrame(interaction_terms, columns=[f"poly_{i}" for i in range(interaction_terms.shape[1])])
test_poly_df = pd.DataFrame(interaction_terms_test, columns=[f"poly_{i}" for i in range(interaction_terms.shape[1])])
train_df = pd.concat([train_df, train_poly_df], axis=1)
test_df = pd.concat([test_df, test_poly_df], axis=1)

# Weather pattern clustering
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
cluster_features = ["pressure", "humidity", "windspeed", "temp_avg", "winddirection_sin", "winddirection_cos"]
train_df["cluster"] = kmeans.fit_predict(train_df[cluster_features])
test_df["cluster"] = kmeans.predict(test_df[cluster_features])

# Create dummy variables for clusters
cluster_dummies_train = pd.get_dummies(train_df["cluster"], prefix="cluster")
cluster_dummies_test = pd.get_dummies(test_df["cluster"], prefix="cluster")
train_df = pd.concat([train_df, cluster_dummies_train], axis=1)
test_df = pd.concat([test_df, cluster_dummies_test], axis=1)

# Prepare data for modeling
features = [col for col in train_df.columns if col not in ["id", "rainfall", "cluster"]]
X = train_df[features]
y = train_df["rainfall"]
X_test = test_df[features]

# Feature selection using RFECV
print("Performing feature selection with RFECV...")
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")
selected_features = [features[i] for i in range(len(features)) if rfecv.support_[i]]
print("Selected features:")
print(selected_features)

X = X[selected_features]
X_test = X_test[selected_features]

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# Handle class imbalance
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning with Optuna
def objective(trial):
    # XGBoost parameters
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    # LightGBM parameters
    lgbm_params = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 150),
        'max_depth': trial.suggest_int('lgbm_max_depth', 3, 10),
        'subsample': trial.suggest_float('lgbm_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }
    
    # CatBoost parameters
    cat_params = {
        'n_estimators': trial.suggest_int('cat_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'random_strength': trial.suggest_float('cat_random_strength', 0.1, 10.0),
        'verbose': 0,
        'random_state': 42
    }
    
    # RandomForest parameters
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
        'random_state': 42
    }
    
    # GradientBoosting parameters
    gb_params = {
        'n_estimators': trial.suggest_int('gb_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('gb_max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('gb_min_samples_split', 2, 10),
        'subsample': trial.suggest_float('gb_subsample', 0.6, 1.0),
        'random_state': 42
    }
    
    # Create models with suggested parameters
    xgb = XGBClassifier(**xgb_params)
    lgbm = LGBMClassifier(**lgbm_params)
    cat = CatBoostClassifier(**cat_params)
    rf = RandomForestClassifier(**rf_params)
    gb = GradientBoostingClassifier(**gb_params)
    
    # Train individual models
    xgb.fit(X_train_resampled, y_train_resampled)
    lgbm.fit(X_train_resampled, y_train_resampled)
    cat.fit(X_train_resampled, y_train_resampled)
    rf.fit(X_train_resampled, y_train_resampled)
    gb.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    xgb_pred = xgb.predict_proba(X_val_scaled)[:, 1]
    lgbm_pred = lgbm.predict_proba(X_val_scaled)[:, 1]
    cat_pred = cat.predict_proba(X_val_scaled)[:, 1]
    rf_pred = rf.predict_proba(X_val_scaled)[:, 1]
    gb_pred = gb.predict_proba(X_val_scaled)[:, 1]
    
    # Create meta-features
    meta_features = np.column_stack([xgb_pred, lgbm_pred, cat_pred, rf_pred, gb_pred])
    
    # Meta-classifier
    meta_clf = LogisticRegression(random_state=42)
    meta_clf.fit(meta_features, y_val)
    
    # Final prediction
    final_pred = meta_clf.predict_proba(meta_features)[:, 1]
    
    # Return ROC AUC score
    return roc_auc_score(y_val, final_pred)

# Optimize hyperparameters
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # Adjust n_trials as needed

print("Best trial:")
trial = study.best_trial
print(f"  ROC AUC: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Create final models with optimized parameters
best_params = study.best_params

# Extract parameters for each model
xgb_best_params = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
lgbm_best_params = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')}
cat_best_params = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}
rf_best_params = {k.replace('rf_', ''): v for k, v in best_params.items() if k.startswith('rf_')}
gb_best_params = {k.replace('gb_', ''): v for k, v in best_params.items() if k.startswith('gb_')}

# Add fixed parameters
xgb_best_params['eval_metric'] = 'logloss'
xgb_best_params['random_state'] = 42
lgbm_best_params['random_state'] = 42
cat_best_params['verbose'] = 0
cat_best_params['random_state'] = 42
rf_best_params['random_state'] = 42
gb_best_params['random_state'] = 42

# Create final models
xgb = XGBClassifier(**xgb_best_params)
lgbm = LGBMClassifier(**lgbm_best_params)
cat = CatBoostClassifier(**cat_best_params)
rf = RandomForestClassifier(**rf_best_params)
gb = GradientBoostingClassifier(**gb_best_params)

# Combine all train and validation data
X_full = np.vstack([X_train_resampled, X_val_scaled])
y_full = np.hstack([y_train_resampled, y_val])

# Create final stacking model
stacked_model = StackingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat), ('rf', rf), ('gb', gb)],
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

# Train final model
print("Training final stacking model...")
stacked_model.fit(X_full, y_full)

# Save the model
joblib.dump(stacked_model, "stacked_model.pkl")

# Feature importance analysis
print("Calculating feature importance...")
feature_importances = {}
for name, est in [('xgb', xgb), ('lgbm', lgbm), ('rf', rf), ('gb', gb)]:
    est.fit(X_train_resampled, y_train_resampled)
    if hasattr(est, 'feature_importances_'):
        feature_importances[name] = est.feature_importances_

# Plot feature importances
plt.figure(figsize=(12, 10))
for i, (name, importances) in enumerate(feature_importances.items()):
    plt.subplot(2, 2, i+1)
    indices = np.argsort(importances)[-15:]  # Top 15 features
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [selected_features[j] for j in indices])
    plt.title(f'Top 15 Feature Importances - {name}')
plt.tight_layout()
plt.show()

# SHAP analysis for XGBoost model
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_train_scaled[:200])  # Sample for faster computation

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train_scaled[:200], feature_names=selected_features)
plt.show()

# Make final predictions
preds_proba = stacked_model.predict_proba(X_test_scaled)[:, 1]
preds = stacked_model.predict(X_test_scaled).astype(int)  # Ensure integer type

# Validate predictions
print(f"Unique prediction values: {np.unique(preds)}")
print(f"Any missing values: {np.isnan(preds).any()}")

# Create submission file according to expected format
submission = test_df[["id"]].copy()
submission["id"] = submission["id"].astype(int)  # Ensure integer type
submission["rainfall"] = preds  # Already integer type
submission.to_csv("stacking_predictions.csv", index=False)
print("Predictions saved to stacking_predictions.csv")

# Check the submission file format
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
print("\nSample submission format:")
print(sample_submission.dtypes)
print("\nOur submission format:")
print(submission.dtypes)

# Verify row count matches
print(f"\nSample submission rows: {len(sample_submission)}")
print(f"Our submission rows: {len(submission)}")

# Create a more detailed output with probabilities (for your reference only)
detailed_submission = test_df[["id"]].copy()
detailed_submission["rainfall"] = preds
detailed_submission["rainfall_probability"] = preds_proba
detailed_submission.to_csv("detailed_predictions.csv", index=False)
print("Detailed predictions saved to detailed_predictions.csv")

# Cross-validation performance
print("Performing cross-validation to estimate model performance...")
cv_scores = cross_val_score(stacked_model, X, y, cv=5, scoring='roc_auc')
print(f"Cross-validation ROC AUC scores: {cv_scores}")
print(f"Mean ROC AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Load and examine the sample submission file
print("\nChecking submission format against sample submission...")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

# Create a submission file that exactly matches the structure of the sample submission
final_submission = pd.DataFrame()
final_submission["id"] = sample_submission["id"]  # Use exact same IDs in same order
final_submission["rainfall"] = preds  # Assign predictions to these IDs

# Verify no missing entries
print(f"Missing values in submission: {final_submission.isnull().sum().sum()}")

# Save the precisely formatted submission
final_submission.to_csv("final_submission.csv", index=False)
print("Final precisely formatted submission saved to final_submission.csv - USE THIS FILE")

# Verify class distribution in final predictions
print(f"Prediction class distribution: {pd.Series(final_submission['rainfall']).value_counts()}")
print(f"Training class distribution: {pd.Series(train_df['rainfall']).value_counts()}")

