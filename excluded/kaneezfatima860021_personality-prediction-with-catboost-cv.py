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


from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold


# 1. Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Keep test IDs for submission
test_ids = test['id']


# 2. Split features/target â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
X = train.drop(['Personality'], axis=1)
y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # 0/1 coding


# 3. Identify categorical columns & fix NaNsÂ â”€â”€â”€â”€â”€â”€â”€â”€
cat_cols = X.select_dtypes('object').columns.tolist()

for col in cat_cols:
    # Cast to string and replace missing values with a placeholder category
    X[col]    = X[col].astype(str).fillna('Missing')
    test[col] = test[col].astype(str).fillna('Missing')

# (Numeric NaNs can stay â€” CatBoost handles them natively)


# 4. Set up StratifiedÂ Kâ€‘Fold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
N_FOLDS, SEED = 5, 42
skf            = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof   = np.zeros(len(train))
preds = np.zeros(len(test))


# 5. Train CatBoost per fold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=4,
        random_state=SEED,
        eval_metric='Accuracy',
        loss_function='Logloss',
        early_stopping_rounds=200,
        verbose=False
    )
    
    model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
        cat_features=cat_cols,
        use_best_model=True
    )
    
    # Outâ€‘ofâ€‘fold predictions
    oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    
    # Testâ€‘set predictions (average)
    preds += model.predict_proba(test)[:, 1] / N_FOLDS
    
    print(f'Fold {fold} âœ”ï¸�')


# 6. OOF accuracy (sanity check) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
oof_labels = (oof > 0.5).astype(int)
oof_acc    = (oof_labels == y).mean()
print(f'\nOOF Accuracy â‰ˆ {oof_acc:.4f}') 


# 7. Build submission â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
test_labels = np.where(preds > 0.5, 'Extrovert', 'Introvert')

submission = pd.DataFrame({'id': test_ids, 'Personality': test_labels})
submission.to_csv('submission.csv', index=False)
print('\nğŸš€ submission.csv saved â€” ready to upload!')




