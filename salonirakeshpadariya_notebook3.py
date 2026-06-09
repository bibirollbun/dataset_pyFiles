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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load your data ---
# Adjust these paths if needed.
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# --- Step 2: Define features and target ---
# Use the columns you want to include. This is an example with 10 original features.
original_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                     'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
target = 'rainfall'

X = train_df[original_features]
y = train_df[target]

# For test data, if you have an "id" column, keep it aside:
test_ids = test_df['id'] if 'id' in test_df.columns else np.arange(len(test_df))
X_test = test_df[original_features]
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)          # X is your training features
X_test_imputed = imputer.transform(X_test)      # X_test is your test features
# --- Step 3: Preprocess features ---
# Here we scale features using StandardScaler. (Adjust or add feature engineering as needed.)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# --- Optional: Split a validation set to gauge performance ---
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 4: Define base models ---
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('lgb', lgb.LGBMClassifier(random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# --- Step 5: Define the meta-model ---
meta_model = LogisticRegression(solver='liblinear', random_state=42)

# The StackingClassifier below uses 5-fold cross-validation on the training set
stacking_clf = StackingClassifier(estimators=base_estimators,
                                  final_estimator=meta_model,
                                  cv=5,
                                  passthrough=True,   # passthrough adds original features to meta-model
                                  n_jobs=-1)

# --- Step 6: Train and evaluate the stacking ensemble ---
stacking_clf.fit(X_train, y_train)
y_val_pred = stacking_clf.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print("Validation AUC:", val_auc)

# If the validation performance looks promising (say, above 0.93),
# retrain the ensemble on all training data and predict on test.
stacking_clf.fit(X_scaled, y)
test_pred = stacking_clf.predict_proba(X_test_scaled)[:, 1]

# --- Step 7: Create submission file ---
submission = pd.DataFrame({'id': test_ids, 'rainfall': test_pred})
submission.to_csv('submission.csv', index=False)
print(submission.head())


