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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Load data
train = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/train.csv')
test = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/test.csv')

# Extract IDs
train_ids = train['id']
test_ids = test['id']

# Targets
y = train['target'].str.replace('Class_', '').astype(int) - 1

# Raw features
X = train.drop(columns=['id', 'target'])
X_test = test.drop(columns=['id'])

# Scaled versions
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Optional: log-transformed
X_log = np.log1p(X)
X_test_log = np.log1p(X_test)

# KFold setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import lightgbm as lgb

# Load original train/test data
train = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/train.csv')
test = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/test.csv')

X = train.drop(columns=['id', 'target'])
y = train['target'].str.replace('Class_', '').astype(int) - 1
X_test = test.drop(columns=['id'])

# Scale and add PCA features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

X_scaled = np.hstack([X_scaled, X_pca])
X_test_scaled = np.hstack([X_test_scaled, X_test_pca])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

meta_train = np.zeros((X.shape[0], 9 * 4))  # 4 models * 9 classes
meta_test = np.zeros((X_test.shape[0], 9 * 4))
model_names = ['lr', 'et', 'nb', 'lgb']

# 1ï¸�âƒ£ Logistic Regression
print("\nğŸ”¸ Training Logistic Regression")
oof_preds = np.zeros((X.shape[0], 9))
test_preds = np.zeros((X_test.shape[0], 9))
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y), 1):
    clf = LogisticRegression(max_iter=2000, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_scaled[train_idx], y[train_idx])
    val_pred = clf.predict_proba(X_scaled[val_idx])
    oof_preds[val_idx] = val_pred
    test_preds += clf.predict_proba(X_test_scaled) / kf.n_splits
    print(f"  ğŸ“˜ Fold {fold}: Log Loss = {log_loss(y[val_idx], val_pred):.4f}, Accuracy = {accuracy_score(y[val_idx], val_pred.argmax(1)):.4f}")
meta_train[:, 0:9] = oof_preds
meta_test[:, 0:9] = test_preds

# 2ï¸�âƒ£ Extra Trees + Calibration
print("\nğŸ”¸ Training Calibrated Extra Trees")
oof_preds = np.zeros((X.shape[0], 9))
test_preds = np.zeros((X_test.shape[0], 9))
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y), 1):
    base_clf = ExtraTreesClassifier(n_estimators=500, max_features=0.4, random_state=fold)
    clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
    clf.fit(X_scaled[train_idx], y[train_idx])
    val_pred = clf.predict_proba(X_scaled[val_idx])
    oof_preds[val_idx] = val_pred
    test_preds += clf.predict_proba(X_test_scaled) / kf.n_splits
    print(f"  ğŸ“˜ Fold {fold}: Log Loss = {log_loss(y[val_idx], val_pred):.4f}, Accuracy = {accuracy_score(y[val_idx], val_pred.argmax(1)):.4f}")
meta_train[:, 9:18] = oof_preds
meta_test[:, 9:18] = test_preds

# 3ï¸�âƒ£ Naive Bayes
print("\nğŸ”¸ Training Naive Bayes with log1p features")
X_nb = np.log1p(X).values
X_test_nb = np.log1p(X_test).values
oof_preds = np.zeros((X.shape[0], 9))
test_preds = np.zeros((X_test.shape[0], 9))
for fold, (train_idx, val_idx) in enumerate(kf.split(X_nb, y), 1):
    clf = MultinomialNB()
    clf.fit(X_nb[train_idx], y[train_idx])
    val_pred = clf.predict_proba(X_nb[val_idx])
    oof_preds[val_idx] = val_pred
    test_preds += clf.predict_proba(X_test_nb) / kf.n_splits
    print(f"  ğŸ“˜ Fold {fold}: Log Loss = {log_loss(y[val_idx], val_pred):.4f}, Accuracy = {accuracy_score(y[val_idx], val_pred.argmax(1)):.4f}")
meta_train[:, 18:27] = oof_preds
meta_test[:, 18:27] = test_preds

# 4ï¸�âƒ£ LightGBM
print("\nğŸ”¸ Training LightGBM")
oof_preds = np.zeros((X.shape[0], 9))
test_preds = np.zeros((X_test.shape[0], 9))
params = {
    'objective': 'multiclass',
    'num_class': 9,
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'metric': 'multi_logloss',
    'verbosity': -1,
    'seed': 42
}
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y), 1):
    dtrain = lgb.Dataset(X_scaled[train_idx], label=y[train_idx])
    dval = lgb.Dataset(X_scaled[val_idx], label=y[val_idx], reference=dtrain)
    model = lgb.train(
    params,
    dtrain,
    num_boost_round=300,
    valid_sets=[dval],
    valid_names=["valid"],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    val_pred = model.predict(X_scaled[val_idx])
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test_scaled) / kf.n_splits
    print(f"  ğŸ“˜ Fold {fold}: Log Loss = {log_loss(y[val_idx], val_pred):.4f}, Accuracy = {accuracy_score(y[val_idx], np.argmax(val_pred, axis=1)):.4f}")
meta_train[:, 27:36] = oof_preds
meta_test[:, 27:36] = test_preds

# Save meta features
columns = [f"{m}_Class_{i+1}" for m in model_names for i in range(9)]
meta_train_df = pd.DataFrame(meta_train, columns=columns)
meta_test_df = pd.DataFrame(meta_test, columns=columns)
meta_test_df.insert(0, 'id', test['id'])

meta_train_df.to_csv("meta_train.csv", index=False)
meta_test_df.to_csv("meta_test_raw.csv", index=False)
print("\nâœ… Saved meta_train.csv and meta_test_raw.csv")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb

# Load meta features and labels
meta_train = pd.read_csv('/kaggle/working/meta_train.csv')
meta_test = pd.read_csv('/kaggle/working/meta_test_raw.csv')
test_ids = meta_test['id']
X_meta = meta_train.values
X_test_meta = meta_test.drop(columns='id').values
y = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/train.csv')['target'].str.replace('Class_', '').astype(int) - 1

# Split meta for validation
X_train, X_val, y_train, y_val = train_test_split(X_meta, y, test_size=0.2, stratify=y, random_state=42)

# DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test_meta)

# Parameters
params = {
    "objective": "multi:softprob",
    "num_class": 9,
    "eval_metric": "mlogloss",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "lambda": 2.0,
    "alpha": 0.8,
    "seed": 42
}

# Cross-validation
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    nfold=10,
    early_stopping_rounds=20,
    stratified=True,
    verbose_eval=50,
    as_pandas=True
)

best_round = cv_results['test-mlogloss-mean'].idxmin()

# Train final model
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=best_round,
    evals=[(dtrain, "train"), (dval, "eval")],
    verbose_eval=False
)

# Predict
val_preds = xgb_model.predict(dval)
val_logloss = log_loss(y_val, val_preds)
val_acc = accuracy_score(y_val, np.argmax(val_preds, axis=1))

print(f"\nğŸ“‰ Validation Log Loss: {val_logloss:.5f}")
print(f"ğŸ�¯ Validation Accuracy: {val_acc:.5f}")

# Test prediction
test_preds = xgb_model.predict(dtest)

# Format for Kaggle
submission = pd.DataFrame(test_preds, columns=[f'Class_{i+1}' for i in range(9)])
submission.insert(0, 'id', test_ids)
submission.to_csv('xgb_level2_preds_FL3.csv', index=False)
print("âœ… Saved: xgb_level2_preds_FL3.csv")



import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
import numpy as np

# Load meta features
meta_train = pd.read_csv("/kaggle/working/meta_train.csv")
meta_test = pd.read_csv("/kaggle/working/meta_test_raw.csv")
test_ids = meta_test['id']
X = meta_train.values
X_test = meta_test.drop(columns='id').values
y = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv")['target'].str.replace('Class_', '').astype(int) - 1

# LightGBM parameters
# params = {
#     'objective': 'multiclass',
#     'num_class': 9,
#     'metric': 'multi_logloss',
#     'learning_rate': 0.01,
#     'max_depth': 10,
#     'num_leaves': 128,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'lambda_l1': 0.5,
#     'lambda_l2': 1.0,
#     'seed': 42
# }
params = {
    'objective': 'multiclass',
    'num_class': 9,
    'metric': 'multi_logloss',
    'learning_rate': 0.01,  # Slightly higher learning rate
    'num_leaves': 128,  # Reduced num_leaves for better generalization
    'max_depth': 10,  # Slightly lower max depth to avoid overfitting
    'min_child_samples': 50,  # Increased min_child_samples to avoid overfitting
    'feature_fraction': 0.8,  # Slightly increased feature_fraction to make use of more features
    'bagging_fraction': 0.85,  # Slightly increased bagging_fraction
    'bagging_freq': 1,
    'max_bin': 512 ,
    'lambda_l1': 0.7,  # Increased L1 regularization to reduce overfitting
    'lambda_l2': 0.8,  # Increased L2 regularization to reduce overfitting
    'min_split_gain': 0.02,  # Slightly increased min_split_gain to make splits more meaningful
    'verbosity': -1,
    'boosting_type': 'gbdt'
}

# Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], 9))
test_preds = np.zeros((X_test.shape[0], 9))

print("ğŸ”„ Training LightGBM Level 2 (fixed 500 rounds)...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nğŸ“‚ Fold {fold + 1}")
    dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
    dval = lgb.Dataset(X[val_idx], label=y[val_idx])
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )
    # model = lgb.train(
    #     params,
    #     lgb_train,
    #     num_boost_round=3000,
    #     valid_sets=[lgb_train, lgb_valid],
    #     valid_names=["train", "valid"],
    #     early_stopping_rounds=100,
    #     verbose_eval=100
    # )

    val_pred = model.predict(X[val_idx])
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / kf.n_splits

    ll = log_loss(y[val_idx], val_pred)
    acc = accuracy_score(y[val_idx], np.argmax(val_pred, axis=1))
    print(f"  ğŸ“˜ Fold {fold}: Log Loss = {ll:.5f}, Accuracy = {acc:.5f}")

# Evaluate full OOF
val_logloss = log_loss(y, oof_preds)
val_acc = accuracy_score(y, np.argmax(oof_preds, axis=1))
print(f"\nğŸ“‰ Full Validation Log Loss: {val_logloss:.5f}")
print(f"ğŸ�¯ Full Validation Accuracy: {val_acc:.5f}")

# Save predictions
submission = pd.DataFrame(test_preds, columns=[f'Class_{i+1}' for i in range(9)])
submission.insert(0, 'id', test_ids)
submission.to_csv("lgb_level2_preds_FL3.csv", index=False)
print("âœ… Saved: lgb_level2_preds_FL3.csv")



from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
import numpy as np

# Load meta features and labels
meta_train = pd.read_csv('/kaggle/working/meta_train.csv')
meta_test = pd.read_csv('/kaggle/working/meta_test_raw.csv')
y = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/train.csv')['target'].str.replace('Class_', '').astype(int) - 1
test_ids = meta_test['id']

X = meta_train.values
X_test = meta_test.drop(columns=['id']).values

# Set up CatBoost parameters
catboost_params = {
    'loss_function': 'MultiClass',  # ğŸ‘ˆ Fix here
    'eval_metric': 'MultiClass',
    'learning_rate': 0.03,
    'depth': 6,
    'iterations': 500,
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 30
}

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((X.shape[0], 9))
test_preds = np.zeros((X_test.shape[0], 9))

print("ğŸ”„ Training CatBoost Level 2...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)

    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict_proba(X_val)
    test_pred = model.predict_proba(X_test)
    
    oof_preds[val_idx] = val_pred
    test_preds += test_pred / kf.n_splits

    ll = log_loss(y_val, val_pred)
    acc = accuracy_score(y_val, np.argmax(val_pred, axis=1))
    print(f"  ğŸ“˜ Fold {fold}: Log Loss = {ll:.5f}, Accuracy = {acc:.5f}")

# Final validation score
val_logloss = log_loss(y, oof_preds)
val_acc = accuracy_score(y, np.argmax(oof_preds, axis=1))
print(f"\nğŸ“‰ Full Validation Log Loss: {val_logloss:.5f}")
print(f"ğŸ�¯ Full Validation Accuracy: {val_acc:.5f}")

# Save predictions for Level 3
catboost_submission = pd.DataFrame(test_preds, columns=[f'Class_{i+1}' for i in range(9)])
catboost_submission.insert(0, 'id', test_ids)
catboost_submission.to_csv("catboost_level2_preds_FL3.csv", index=False)
print("âœ… Saved: catboost_level2_preds_FL3.csv")



import pandas as pd

# Load Level 2 predictions
xgb_preds = pd.read_csv('/kaggle/working/xgb_level2_preds_FL3.csv')
cat_preds = pd.read_csv('/kaggle/working/catboost_level2_preds_FL3.csv')
lgb_preds = pd.read_csv('/kaggle/working/lgb_level2_preds_FL3.csv')  # New model prediction

# Make sure the ID columns match
assert (xgb_preds['id'] == cat_preds['id']).all()
assert (xgb_preds['id'] == lgb_preds['id']).all()

# Drop ID column for averaging
ids = xgb_preds['id']
xgb_probs = xgb_preds.drop(columns=['id']).values
cat_probs = cat_preds.drop(columns=['id']).values
lgb_probs = lgb_preds.drop(columns=['id']).values

# Weighted average (you can tune the weights)
final_probs = 0.5 * xgb_probs + 0.3 * cat_probs + 0.2 * lgb_probs  # Adjust weights accordingly

# Create final submission
final_submission = pd.DataFrame(final_probs, columns=[f'Class_{i}' for i in range(1, 10)])
final_submission.insert(0, 'id', ids)

# Save submission file
final_submission.to_csv('final_submission_level3_FL3.csv', index=False)

print("âœ… Level 3 blending complete â€” final_submission_level3_FL3.csv ready!")
submission=pd.read_csv("/kaggle/working/final_submission_level3_FL3.csv")
submission.head(10)


