# ğŸ“¦ Step 1: Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ğŸ“‚ Step 2: Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# ğŸ§¼ Step 3: Preprocess
target = 'y'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

# Encode categorical columns
cat_cols = X.select_dtypes('object').columns
X = pd.get_dummies(X, columns=cat_cols)
X_test = pd.get_dummies(X_test, columns=cat_cols)

# Align train and test
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# ğŸ“Š Step 4: Training with Stratified K-Fold CV
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f'ğŸ“� Fold {fold+1}')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=1000,
        objective='binary',
        boosting_type='gbdt',
        learning_rate=0.01,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbosity=-1,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=100)
        ]
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits

# ğŸ“ˆ Step 5: Evaluate ROC AUC
cv_auc = roc_auc_score(y, oof_preds)
print(f"\nâœ… Cross-validated AUC: {cv_auc:.5f}")

# ğŸ“� Step 6: Create Submission File
submission = sample_submission.copy()
submission['y'] = test_preds
submission.to_csv('submission.csv', index=False)
print("ğŸ“¦ Submission file saved as submission.csv")


