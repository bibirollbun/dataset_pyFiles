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


# âœ… 1. IMPORTS
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# âœ… 2. LOAD DATA
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Columns:", train.columns.tolist())

# âœ… 3. PREPARE FEATURES
target = train['y']
train = train.drop(['id', 'y'], axis=1)
test_ids = test['id']
test = test.drop(['id'], axis=1)

# âœ… 4. ENCODE CATEGORICALS
cat_cols = train.select_dtypes(include='object').columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    le_dict[col] = le

# âœ… 5. TRAIN WITH STRATIFIED K-FOLD
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
cv_scores = []

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(folds.split(train, target)):
    print(f"\nðŸ§  Fold {fold+1}")
    
    X_train, y_train = train.iloc[train_idx], target.iloc[train_idx]
    X_val, y_val = train.iloc[val_idx], target.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        objective='binary',
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc'
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    score = roc_auc_score(y_val, val_pred)
    cv_scores.append(score)
    
    test_preds += model.predict_proba(test)[:, 1] / folds.n_splits

print("\nâœ… CV AUC Scores:", cv_scores)
print(f"âœ… Mean CV AUC: {np.mean(cv_scores):.5f}")

# âœ… 6. SUBMIT
submission['y'] = test_preds
submission['y'] = submission['y'].clip(0.0001, 0.9999)  # avoid perfect 0s and 1s
submission.to_csv('submission.csv', index=False)
submission.head()


