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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# 1. Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 2. Encode Categoricals
cat_features = ['Soil Type', 'Crop Type']
for col in cat_features:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# 3. Encode Target
le = LabelEncoder()
train['Fertilizer Name Enc'] = le.fit_transform(train['Fertilizer Name'])
target = 'Fertilizer Name Enc'

# 4. Prepare Features (drop id, target)
features = [
    'Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
    'Nitrogen', 'Potassium', 'Phosphorous'
]
X = train[features]
y = train[target]
X_test = test[features]

# 5. Local Validation (hold out a local set)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# 6. Train LightGBM
params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'verbosity': -1,
    'seed': 42,
}
model = lgb.LGBMClassifier(**params, n_estimators=200)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)])

# 7. Predict top3 on validation set for local MAP@3 calculation
val_prob = model.predict_proba(X_val)
val_top3 = np.argsort(val_prob, axis=1)[:, -3:][:, ::-1]
val_preds = [le.inverse_transform(row) for row in val_top3]
val_true = le.inverse_transform(y_val)

# 8. MAP@3 function
def mapk(actual, predicted, k=3):
    """
    actual: list-like of ground truth labels (strings)
    predicted: list of list of predictions (top k for each sample, as strings)
    """
    score = 0.0
    for a, p in zip(actual, predicted):
        try:
            score += 1.0 / (p.tolist().index(a) + 1) if a in p[:k] else 0.0
        except ValueError:
            score += 0.0
    return score / len(actual)

# 9. Print local MAP@3
print("Local MAP@3:", mapk(val_true, val_preds, k=3))

# 10. Retrain on all data for final submission
model_final = lgb.LGBMClassifier(**params, n_estimators=200)
model_final.fit(X, y)

# 11. Predict on test set
test_prob = model_final.predict_proba(X_test)
test_top3 = np.argsort(test_prob, axis=1)[:, -3:][:, ::-1]
test_preds = [le.inverse_transform(row) for row in test_top3]
test_preds_str = [' '.join(row) for row in test_preds]

# 12. Submission
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': test_preds_str
})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

