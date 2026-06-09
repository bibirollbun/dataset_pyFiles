# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


df_train.shape,df_test.shape


df_train.info()


df_train.isnull().sum()


df_test.isnull().sum()


df_train.head(3)



# List of categorical columns to encode
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Create a copy of your dataframe
df_encoded = df_train.copy()

# Initialize LabelEncoder
le = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    df_test[col] = le.fit_transform(df_test[col])

# Now df_encoded has label encoded categorical features



df_test.head(3)


df_encoded.info()




# Features to scale
features_to_scale = ['age', 'balance', 'duration']

# Create a copy of your dataframe
df_scaled = df_encoded.copy()

# Initialize scaler
scaler = StandardScaler()

# Fit and transform the selected features
df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
df_test[features_to_scale] = scaler.fit_transform(df_test[features_to_scale])


df_scaled.head(3)


X = df_scaled.drop('y',axis =1)  # Drop target column
y = df_scaled['y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer



def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }

    model = XGBClassifier(**params)
    
    # Cross-validation for robust results
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(accuracy_score))
    
    return scores.mean()


def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', -1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }

    model = LGBMClassifier(**params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(accuracy_score))
    
    return scores.mean()


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=30)
print("Best XGBoost Params:", study_xgb.best_params)
print("Best Accuracy:", study_xgb.best_value)

# For LightGBM
study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=30)
print("Best LightGBM Params:", study_lgbm.best_params)
print("Best Accuracy:", study_lgbm.best_value)


best_xgb = XGBClassifier(**study_xgb.best_params)
best_xgb.fit(X_train, y_train)

best_lgbm = LGBMClassifier(**study_lgbm.best_params)
best_lgbm.fit(X_train, y_train)



from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier

models = {
    'XGBoost': XGBClassifier(
        **study_xgb.best_params,
        use_label_encoder=False,   # keep required
        eval_metric='mlogloss',
        random_state=42
    ),
    
    'LightGBM': LGBMClassifier(
        **study_lgbm.best_params,
        random_state=42,
        verbose=-1
    )
}




from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[(name, clf) for name, clf in models.items()],
    voting='soft',    # Soft voting uses probability averaging
    n_jobs=-1         # Use all CPU cores
)



# --- Cross Validation ---
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

auc_scores = []  # Store AUC-ROC for each fold
all_preds = []   # Store predictions for test set from each fold

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    print(f"Training fold {fold}...")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train the ensemble
    ensemble.fit(X_train, y_train)

    # Predict probabilities for validation
    y_val_proba = ensemble.predict_proba(X_val)[:, 1]

    # Predict probabilities for the test set
    test_proba = ensemble.predict_proba(df_test)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_val, y_val_proba)
    auc_scores.append(auc)

    # Save predictions for later averaging
    all_preds.append(test_proba)

# --- Accuracy using cross_val_score ---
accuracy = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy').mean()

# --- Results ---
print(f"\nAUC-ROC per fold: {auc_scores}")
print(f"Mean AUC-ROC: {np.mean(auc_scores):.4f}")
print(f"Mean Accuracy: {accuracy:.4f}")




sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


final_preds = np.mean(all_preds, axis=0)

# --- Submission ---
submission = pd.DataFrame({
    'id': sample_submission.id,
    'y': final_preds
})
submission.to_csv('submission_ensemble.csv', index=False)
print(submission.head())

