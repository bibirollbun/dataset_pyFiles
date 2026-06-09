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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import optuna
from sklearn.impute import SimpleImputer


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").rename(columns={'temparature': 'temperature'})
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").rename(columns={'temparature': 'temperature'})

# Separate features and target
X = train.drop(columns=["id", "rainfall"])
y = train["rainfall"]
X_test = test.drop(columns=["id"])

# Handle missing values
numeric_imputer = SimpleImputer(strategy='median')
X_imputed = numeric_imputer.fit_transform(X)
X_test_imputed = numeric_imputer.transform(X_test)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# Apply PCA to reduce dimensions
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Reduced features from {X_scaled.shape[1]} to {X_pca.shape[1]}")


from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import optuna

def cat_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'task_type': 'GPU',  # Use CPU if GPU is unavailable
        'eval_metric': 'AUC',
        'random_state': 42,
        'verbose': False,
        'early_stopping_rounds': 50  # Early stopping
    }
    
    model = CatBoostClassifier(**params)
    scores = []
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced to 3 folds
    for train_idx, val_idx in kf.split(X_pca, y):
        X_train_fold, X_val_fold = X_pca[train_idx], X_pca[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Use eval_set for early stopping
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), use_best_model=True)
        
        preds = model.predict_proba(X_val_fold)[:, 1]
        scores.append(roc_auc_score(y_val_fold, preds))
    
    return np.mean(scores)

cat_study = optuna.create_study(direction='maximize')
cat_study.optimize(cat_objective, n_trials=50)
cat_best_params = cat_study.best_params
print("Best CatBoost Parameters:", cat_best_params)



# Train the final model
final_model = CatBoostClassifier(**cat_best_params, verbose=False)
final_model.fit(X_pca, y)


# Generate predictions
test_preds = final_model.predict_proba(X_test_pca)[:, 1]


submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": test_preds
})
submission.to_csv("submission_catboost.csv", index=False)

print("Submission file created successfully!")


submission.head()


from IPython.display import FileLink

# Create a download link for the file
FileLink('submission_catboost.csv')

