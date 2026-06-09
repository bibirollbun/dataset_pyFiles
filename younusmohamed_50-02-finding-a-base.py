# !pip install scikeras --quiet


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, time

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings("ignore")

# Use seaborn style
plt.style.use('seaborn-whitegrid')
%matplotlib inline

import time
start_time = time.time()


# Load datasets and handle missing values
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Submission shape:", submission.shape)

# Impute the missing winddirection in test with median
if test['winddirection'].isnull().sum() > 0:
    test['winddirection'].fillna(test['winddirection'].median(), inplace=True)

# Define feature set and target
X = train.drop(columns=['id', 'rainfall'])
y = train['rainfall']
X_test = test.drop(columns=['id'])
test_ids = test['id']

end_time = time.time()
print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")


# Define various preprocessing pipelines
preproc_pipelines = {
    'raw': None,  # No scaling, using raw features
    'standard': StandardScaler(),
    'robust': RobustScaler(),
    'minmax': MinMaxScaler(),
    'pca': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95))])
}


# Define model dictionary and hyperparameter grids

# Logistic Regression (with L2 & L1)
models = {}
param_grids = {}

# Logistic Regression variants (using logistic regression for binary classification)
models['logistic'] = LogisticRegression(solver='liblinear', max_iter=1000)
param_grids['logistic'] = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Decision Tree
models['dt'] = DecisionTreeClassifier(random_state=42)
param_grids['dt'] = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Random Forest
models['rf'] = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grids['rf'] = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

# Extra Trees
models['et'] = ExtraTreesClassifier(random_state=42, n_jobs=-1)
param_grids['et'] = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

# Gradient Boosting (sklearn)
models['gb'] = GradientBoostingClassifier(random_state=42)
param_grids['gb'] = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7]
}

# XGBoost (using GPU - tree_method='gpu_hist' if available)
models['xgb'] = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grids['xgb'] = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'colsample_bytree': [0.5, 0.7, 1.0]
}

# LightGBM (with GPU support)
models['lgb'] = lgb.LGBMClassifier(device='gpu', random_state=42)
param_grids['lgb'] = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7]
}

# CatBoost (with GPU support; silent mode to reduce output)
# models['cat'] = CatBoostClassifier(task_type='GPU', verbose=0, random_state=42)
param_grids['cat'] = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 5, 7]
}

# Deep Neural Network (Keras)
def create_dnn_model(optimizer='adam', dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X.shape[1]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC'])
    return model

# models['dnn'] = KerasClassifier(build_fn=create_dnn_model, verbose=0)
param_grids['dnn'] = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.2, 0.3],
    'epochs': [20],
    'batch_size': [32, 64]
}

end_time = time.time()
print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")


# Train models over different preprocessing pipelines and generate submissions

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create a directory to store submissions
os.makedirs("submissions", exist_ok=True)

for prep_name, scaler in preproc_pipelines.items():
    # Preprocess training data
    if scaler is not None:
        X_train_trans = scaler.fit_transform(X)
        X_test_trans = scaler.transform(X_test)
    else:
        X_train_trans = X.values  # raw
        X_test_trans = X_test.values
        
    for model_name, model in models.items():
        print(f"Processing: Preprocessor = {prep_name}, Model = {model_name}")
        param_grid = param_grids[model_name]
        
        # Create pipeline (if needed, we can combine scaler and model but here we already preprocessed)
        # Use RandomizedSearchCV for hyperparameter tuning
        search = RandomizedSearchCV(model, param_grid, n_iter=5, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
        start = time.time()
        search.fit(X_train_trans, y)
        elapsed = time.time() - start
        best_score = search.best_score_
        best_estimator = search.best_estimator_
        
        print(f"Best CV ROC AUC: {best_score:.4f} in {elapsed:.1f} sec")
        
        # Generate test predictions (probability for class 1)
        test_preds = best_estimator.predict_proba(X_test_trans)[:,1]
        sub_df = pd.DataFrame({'id': test_ids, 'rainfall': test_preds})
        sub_filename = f"submissions/{prep_name}_{model_name}_submission.csv"
        sub_df.to_csv(sub_filename, index=False)
        print(f"Submission file saved: {sub_filename}\n")
        
        # Store results for later comparison
        results.append({
            'preproc': prep_name,
            'model': model_name,
            'cv_auc': best_score,
            'estimator': best_estimator,
            'submission_file': sub_filename
        })

        end_time = time.time()
        print(f"Total Execution Time: {end_time - start_time:.2f} seconds\n")


# Compare results and extract feature importance for top 3 models
results_df = pd.DataFrame(results).sort_values(by='cv_auc', ascending=False)
print("Top 3 Models:")
print(results_df[['preproc', 'model', 'cv_auc', 'submission_file']].head(3))

top3 = results_df.head(3)

for idx, row in top3.iterrows():
    est = row['estimator']
    model_name = row['model']
    prep = row['preproc']
    print(f"\nFeature importance for {prep}_{model_name}:")
    
    # For tree-based and linear models we try to extract feature importance/coefficients
    if hasattr(est, 'feature_importances_'):
        importances = est.feature_importances_
        feature_names = X.columns
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
        print(imp_df.head(10))
    elif hasattr(est, 'coef_'):
        coefs = est.coef_.flatten()
        feature_names = X.columns
        coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs}).sort_values(by='coefficient', key=abs, ascending=False)
        print(coef_df.head(10))
    else:
        print("Feature importance not available for this model.")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds\n")


plt.figure(figsize=(12, 6))
# Create a barplot with preprocessing pipelines on the x-axis, and AUC scores as the height; hue = model
sns.barplot(x='preproc', y='cv_auc', hue='model', data=results_df)
plt.title("Comparison of CV ROC AUC Scores by Preprocessing and Model")
plt.xlabel("Preprocessing Pipeline")
plt.ylabel("CV ROC AUC")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Load the individual submission files
robust_sub = pd.read_csv("submissions/robust_logistic_submission.csv")
minmax_sub = pd.read_csv("submissions/minmax_logistic_submission.csv")
standard_sub = pd.read_csv("submissions/standard_logistic_submission.csv")

# Given CV ROC AUC scores for the three submissions
score_robust = 0.86055
score_minmax = 0.86028
score_standard = 0.85974

# Compute normalized weights (higher score gets slightly higher weight)
total_score = score_robust + score_minmax + score_standard
w_robust = score_robust / total_score
w_minmax = score_minmax / total_score
w_standard = score_standard / total_score

print("Weights:")
print(f"Robust: {w_robust:.4f}, MinMax: {w_minmax:.4f}, Standard: {w_standard:.4f}")

# Compute weighted ensemble predictions
ensemble_pred = (w_robust * robust_sub['rainfall'] +
                 w_minmax * minmax_sub['rainfall'] +
                 w_standard * standard_sub['rainfall'])

# Create final submission DataFrame (assuming the id columns match)
ensemble_sub = pd.DataFrame({'id': robust_sub['id'], 'rainfall': ensemble_pred})

# Save the weighted ensemble submission as 'submission.csv'
ensemble_sub.to_csv("submission.csv", index=False)
print("Weighted ensemble submission saved as submission.csv")

