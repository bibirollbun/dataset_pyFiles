import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Data imports
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

y = train['diagnosed_diabetes']
test_ids = test['id']

cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

# Encode categorical features
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

feature_cols = [c for c in train.columns if c not in ['diagnosed_diabetes', 'id']]
X, X_test = train[feature_cols], test[feature_cols]

print(f"Using {len(feature_cols)} raw features")

# LightGBM parameters
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.01, 'num_leaves': 15, 'max_depth': 4,
    'min_child_samples': 200, 'feature_fraction': 0.5, 'bagging_fraction': 0.5,
    'bagging_freq': 5, 'reg_alpha': 2.0, 'reg_lambda': 2.0, 'verbose': -1, 'seed': 42
}

N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof, preds = np.zeros(len(X)), np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.train(
        lgb_params,
        lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx]),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])],
        callbacks=[lgb.early_stopping(300, verbose=False)]
    )
    oof[val_idx] = model.predict(X.iloc[val_idx])
    preds += model.predict(X_test) / N_FOLDS
    print(f"Fold {fold+1}: {roc_auc_score(y.iloc[val_idx], oof[val_idx]):.5f}, trees: {model.best_iteration}")

print(f"\nCV: {roc_auc_score(y, oof):.5f}")

# Save submission to /kaggle/working/
submission = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Saved submission.csv to /kaggle/working/")


