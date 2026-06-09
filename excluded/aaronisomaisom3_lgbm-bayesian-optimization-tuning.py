# Kaggle Rainfall Prediction - LightGBM with Bayesian Optimization
# Author: Aaron Isom

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import hp
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.ensemble import BaggingClassifier
import warnings

warnings.filterwarnings('ignore')

# Load Data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')

# Display first few rows
display(train_df.head(10))
display("Train Shape", train_df.shape)
display("Test Shape", test_df.shape)

# Describe the data
display(train_df.describe())

# Display information about dtypes and missing values
display("Train Data Info:", train_df.info())

# Check target distribution
display("Target Distribution:", train_df['rainfall'].value_counts(normalize=True))

# Missing values
display("Train Missing Values:", train_df.isnull().sum().sum())
display("Test Missing Values:", test_df.isnull().sum().sum())

# Fix missing values in Test
test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].mean())

# Separate features and target
X = train_df.drop(columns=['rainfall'])
#test_df = test_df.drop(columns=['day'])
y = train_df['rainfall']

# Standardize the selected features (optional, helps with LightGBM stability)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_df)

X_train, X_holdout, y_train, y_holdout = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Check class counts
class_counts = y.value_counts()
print("Class Distribution:\n", class_counts)

# Compute imbalance ratio
imbalance_ratio = class_counts.min() / class_counts.max()
print(f"\nImbalance Ratio: {imbalance_ratio:.4f}")  # Closer to 0 means more imbalance

# Plot imbalances
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.xlabel("Class Labels")
plt.ylabel("Frequency")
plt.title("Class Distribution in Rainfall Training Data")
plt.xticks([0, 1], labels=["No Rain", "Rain"])  # Adjust labels if needed
plt.show()

# Determine the class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
scale_weight = (class_weights[1] / class_weights[0])
print("Computed Class Weights:", scale_weight)

# Evaluation Function for Bayesian Optimization
def lgb_eval(max_depth, num_leaves, subsample, colsample_bytree, learning_rate, reg_alpha, reg_lambda, min_child_samples, 
             bagging_fraction, feature_fraction, min_split_gain, scale_pos_weight=0.327, n_estimators=500,):
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'dart',
        'max_bin': 1024,
        'min_gain_to_split': 0.3,
        'n_estimators': int(n_estimators),
        'scale_pos_weight': scale_pos_weight,
        'max_depth': int(max_depth),
        'num_leaves': int(num_leaves),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'learning_rate': learning_rate,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'min_child_samples': int(min_child_samples),
        'bagging_fraction': bagging_fraction,
        'feature_fraction': feature_fraction,
        'min_split_gain': min_split_gain,
        'silent': False,
        'verbose': -1,
    }
    
    # Fix: Create Dataset with max_bin before training
    train_data = lgb.Dataset(X_train, label=y_train)  
    
    # Train model using cross-validation
    cv_results = lgb.cv(params, train_data, num_boost_round=300, nfold=5, stratified=True, shuffle=True, seed=42)

    # Return best AUC score
    return max(cv_results['valid auc-mean'])  # Maximize AUC

#Bayesian Optimization
optimizer = BayesianOptimization(f=lgb_eval, pbounds={
                                'max_depth': (3, 10),
                                'num_leaves': (25, 100),
                                'subsample': (0.5, 1.0),
                                'colsample_bytree': (0.5, 0.9),
                                'learning_rate': (0.02, 0.1),
                                'reg_alpha': (0.01, 10),
                                'reg_lambda': (0.01, 10),
                                'min_child_samples': (10, 75),
                                'bagging_fraction': (0.2, 1.0),
                                'feature_fraction': (0.2, 1.0),
                                'min_split_gain': (0, 0.9)
                                 },
                                 random_state=42, 
                                 verbose=-1)
        
# Start Bayesian Optimization
optimizer.maximize(init_points=50, n_iter=150)
print("Final result:", optimizer.max)

# Get best parameters
best_params = optimizer.max['params']
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_child_samples'] = int(best_params['min_child_samples'])
#best_params['scale_pos_weight'] = round(best_params['scale_pos_weight'], 2)
display("Best params:", best_params)

# Train the final model
final_model = lgb.LGBMClassifier(**best_params, callbacks=[lgb.early_stopping(stopping_rounds=100)], random_state=42, verbose=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))

for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_vl = X_train[train_idx], X_train[val_idx]
    y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

    final_model.fit(X_tr, y_tr)
    oof_preds[val_idx] = final_model.predict_proba(X_vl)[:, 1]

# AUC on out-of-fold predictions
roc_score = roc_auc_score(y_train, oof_preds)

final_model.fit(X_scaled, y)

# Predict on holdout set
y_holdout_pred = final_model.predict(X_holdout)
y_holdout_prob = final_model.predict_proba(X_holdout)[:, 1]

# Metrics
acc = accuracy_score(y_holdout, y_holdout_pred)
f1 = f1_score(y_holdout, y_holdout_pred)
auc = roc_auc_score(y_holdout, y_holdout_prob)

print(f"Holdout Accuracy: {acc:.4f}")
print(f"Holdout F1 Score: {f1:.4f}")
print(f"Holdout ROC-AUC: {auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_holdout, y_holdout_prob)
plt.plot(fpr, tpr, label=f"Holdout ROC (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Holdout Set")
plt.legend()
plt.show()


# Prepare submission file
test_preds = final_model.predict_proba(X_test_scaled)[:, 1]
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = test_preds
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")
display(submission)


# Remove old file(s)
# import os
# os.remove('/kaggle/working/submission.csv')

