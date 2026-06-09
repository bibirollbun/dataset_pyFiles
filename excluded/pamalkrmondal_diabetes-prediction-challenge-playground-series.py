!pip install pandas>=2.0.0 numpy>=1.20.0 scikit-learn>=1.0.0 lightgbm>=4.0.0 matplotlib seaborn optuna



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score

# Configuration
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')



# Define file paths
TRAIN_PATH = '/kaggle/input/playground-series-s5e12/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e12/test.csv'
SUB_PATH = '/kaggle/input/playground-series-s5e12/sample_submission.csv'

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
submission_df = pd.read_csv(SUB_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
display(train_df.head())



# Check target distribution
print("Target Distribution:")
print(train_df['diagnosed_diabetes'].value_counts(normalize=True))

# Check info for data types and missing values
print("\nData Info:")
train_df.info()



# Combine for consistent encoding
all_data = pd.concat([train_df.drop('diagnosed_diabetes', axis=1), test_df], axis=0)

# Feature Engineering: Pulse Pressure
all_data['pulse_pressure'] = all_data['systolic_bp'] - all_data['diastolic_bp']

# Identify categorical columns
cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

# Label Encoding
le = LabelEncoder()
for col in cat_cols:
    # Fill missing with 'Unknown' before encoding if any exist (robustness)
    all_data[col] = all_data[col].astype(str).fillna('Unknown')
    all_data[col] = le.fit_transform(all_data[col])

# Split back
X = all_data.iloc[:len(train_df)].drop('id', axis=1)
X_test = all_data.iloc[len(train_df):].drop('id', axis=1)
y = train_df['diagnosed_diabetes']

print("Processed Feature Matrix Shape:", X.shape)



def objective(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': 42,
        'verbose': -1
    }
    
    # Use a smaller CV for speed during tuning
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
        
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("\nBest Params:")
print(study.best_params)



# Use Best Params from Optuna (or fall back to a robust config)
best_params = study.best_params.copy()

# Update specific training controls for the final run
best_params.update({
    'n_estimators': 5000,          # Allow model to learn longer
    'learning_rate': 0.01,         # Slower learning rate for better convergence
    'random_state': 42,
    'verbose': -1
})

# Initialize KFold (5 splits for robust final evaluation)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))
scores = []

print("Starting Final Training with Early Stopping...")

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(**best_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(stopping_rounds=100, verbose=False), log_evaluation(period=500)]
    )
    
    # Predict probabilities
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / folds.get_n_splits()
    
    score = roc_auc_score(y_val, val_pred)
    scores.append(score)
    print(f"Fold {fold+1} AUC: {score:.4f}")

print(f"\nMean AUC: {np.mean(scores):.4f}")



# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_preds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")
display(submission.head())


