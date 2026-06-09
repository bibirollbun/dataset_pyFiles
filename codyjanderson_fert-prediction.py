# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import gc

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifierCV

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train.head()


test.head()


train.isnull().sum()
train.duplicated().sum()


train['Fertilizer Name'].value_counts().plot(kind='bar', title='Fertilizer Name Distribution')


for col in ['Soil Type', 'Crop Type']:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=train, order=train[col].value_counts().index)
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for col in ['Soil Type', 'Crop Type']:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, hue='Fertilizer Name', data=train)
    plt.title(f'Fertilizer Name by {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


train[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']].hist(
    bins=20, figsize=(16, 8), layout=(2, 3))


# Step 1: Separate features and target
X = train.drop(columns=["Fertilizer Name"])
y = train["Fertilizer Name"]

# Step 2: Label encode the target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Step 3: Only label encode if column is object/string or has low cardinality
feature_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object' or X[col].nunique() < 100:  # Adjust threshold if needed
        le_feat = LabelEncoder()
        X[col] = le_feat.fit_transform(X[col])
        test[col] = le_feat.transform(test[col])
        feature_encoders[col] = le_feat
    else:
        # Leave numeric features as-is
        X[col] = X[col]
        test[col] = test[col]

# Step 4: Clean up
del y
gc.collect()

# Step 5: Stratified K-Fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def apk(actual, predicted, k=3):
    pred_list = list(predicted[:k])
    if actual in pred_list:
        return 1 / (pred_list.index(actual) + 1)
    return 0

def mapk(actual, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# Helper: Get Top-3 Predictions
def selecting_top_3(pred_probs):
    return np.argsort(pred_probs, axis=1)[:, -3:][:, ::-1]


# Random Forest Hyperparameters
rf_params = {
    'n_estimators': 300,
    'max_depth': 7,
    'random_state': 42,
    'n_jobs': -1
}

# K-Fold Training Loop
scores, test_preds = [], []

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    rf_md = RandomForestClassifier(**rf_params)
    rf_md.fit(X_train, y_train)

    # Predict probabilities + top-3 classes
    y_proba = rf_md.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)

    # Compute MAP@3
    score = mapk(y_val, top_3_preds, k=3)
    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    # Predict test set
    test_preds.append(rf_md.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

rf_cv_mean = np.mean(scores)
rf_cv_sd = np.std(scores)
print(f"\nRandom Forest CV MAP@3: {rf_cv_mean:.4f} Â± {rf_cv_sd:.4f}")


pred_agg = 0
for i in range(len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('1_sub.csv')

del submission, test_preds
gc.collect()


xgb_params = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'min_child_weight': 50,
    'n_estimators': 300,
    'early_stopping_rounds': 50,
    'n_jobs': -1,
    'random_state': 42
}

# Train XGBoost with Stratified K-Fold
scores, xgb_test_preds = [], []

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    xgb_md = XGBClassifier(**xgb_params)
    xgb_md.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_proba = xgb_md.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    xgb_test_preds.append(xgb_md.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

xgb_cv_mean = np.mean(scores)
xgb_cv_sd = np.std(scores)
print(f"\nXGBoost CV MAP@3: {xgb_cv_mean:.4f} Â± {xgb_cv_sd:.4f}")


pred_agg = 0
for i in range(len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('2_sub.csv')

del submission, test_preds
gc.collect()


# Define CatBoost parameters
cat_params = {
    'iterations': 200,
    'depth': 3,
    'learning_rate': 0.05,
    'loss_function': 'MultiClass',
    'eval_metric': 'TotalF1',
    'early_stopping_rounds': 50,
    'random_seed': 42,
    'verbose': False,
    'task_type': 'CPU'
}

# Train CatBoost with Stratified K-Fold
scores, test_preds = [], []

cat_features = [X.columns.get_loc(col) for col in feature_encoders.keys()]

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    cb_md = CatBoostClassifier(**cat_params)
    cb_md.fit(X_train, y_train,
              cat_features=cat_features,
              eval_set=(X_val, y_val))

    y_proba = cb_md.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    test_preds.append(cb_md.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

cb_cv_mean = np.mean(scores)
cb_cv_sd = np.std(scores)
print(f"\nCatBoost CV MAP@3: {cb_cv_mean:.4f} Â± {cb_cv_sd:.4f}")


pred_agg = 0
for i in range(len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('3_sub.csv')

del submission, test_preds
gc.collect()


# LightGBM parameters
lgb_params = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.05,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_jobs': -1
}

# Train LightGBM with Stratified K-Fold
scores, lgb_test_preds = [], []

cat_features = [X.columns.get_loc(col) for col in feature_encoders.keys()]

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    lgb_md = LGBMClassifier(**lgb_params)
    lgb_md.set_params(early_stopping_rounds=50, verbose=-1)

    lgb_md.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               eval_metric='multi_logloss',
               categorical_feature=cat_features)

    y_proba = lgb_md.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    lgb_test_preds.append(lgb_md.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

lgb_cv_mean = np.mean(scores)
lgb_cv_sd = np.std(scores)
print(f"\nLightGBM CV MAP@3: {lgb_cv_mean:.4f} Â± {lgb_cv_sd:.4f}")


pred_agg = 0
for i in range(len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('4_sub.csv')

del submission, test_preds
gc.collect()


# Logistic Regression parameters
lr_model = OneVsRestClassifier(
    LogisticRegression(C=1.0, 
                       solver='lbfgs', 
                       multi_class='ovr', 
                       max_iter=300, 
                       random_state=42)
)

# Train with Stratified K-Fold
scores, test_preds = [], []

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    lr_model.fit(X_train, y_train)

    y_proba = lr_model.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    test_preds.append(lr_model.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

lr_cv_mean = np.mean(scores)
lr_cv_sd = np.std(scores)
print(f"\nLogistic Regression CV MAP@3: {lr_cv_mean:.4f} Â± {lr_cv_sd:.4f}")


pred_agg = 0
for i in range(len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('5_sub.csv')

del submission, test_preds
gc.collect()


# ExtraTrees parameters
et_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)

# Train with Stratified K-Fold
scores, test_preds = [], []

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    et_model.fit(X_train, y_train)

    y_proba = et_model.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    test_preds.append(et_model.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

et_cv_mean = np.mean(scores)
et_cv_sd = np.std(scores)
print(f"\nExtraTrees CV MAP@3: {et_cv_mean:.4f} Â± {et_cv_sd:.4f}")


pred_agg = 0
for i in range(len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('6_sub.csv')

del submission, test_preds
gc.collect()


# MLP parameters 
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)

# Train with Stratified K-Fold
scores, test_preds = [], []

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    mlp_model.fit(X_train, y_train)

    y_proba = mlp_model.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    test_preds.append(mlp_model.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

mlp_cv_mean = np.mean(scores)
mlp_cv_sd = np.std(scores)
print(f"\nMLPClassifier CV MAP@3: {mlp_cv_mean:.4f} Â± {mlp_cv_sd:.4f}")


pred_agg = 0
for i in range(len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('7_sub.csv')

del submission, test_preds
gc.collect()


# Define the RidgeClassifierCV model
ridge_model = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0], cv=3)

# Train with Stratified K-Fold
scores, test_preds = [], []

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    ridge_model.fit(X_train, y_train)

    # RidgeClassifier doesn't have predict_proba, so we fake it using decision_function
    decision_scores = ridge_model.decision_function(X_val)
    y_proba = decision_scores - decision_scores.min(axis=1, keepdims=True)  
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    test_scores = ridge_model.decision_function(test)
    test_proba = test_scores - test_scores.min(axis=1, keepdims=True)
    test_preds.append(test_proba)

    del X_train, X_val, y_train, y_val
    gc.collect()

ridge_cv_mean = np.mean(scores)
ridge_cv_sd = np.std(scores)
print(f"\nRidgeClassifierCV MAP@3: {ridge_cv_mean:.4f} Â± {ridge_cv_sd:.4f}")


pred_agg = 0
for i in range(len(test_preds)):
    pred_agg += test_preds[i] / len(test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('8_sub.csv')

del submission, test_preds
gc.collect()


# Blend weights 
xgb_weight = 0.4 # second best prediction
lgb_weight = 0.6 # best prediction

# Combine test predictions from each model
ensemble_pred = (
    xgb_weight * np.mean(xgb_test_preds, axis=0) +
    lgb_weight * np.mean(lgb_test_preds, axis=0)
)

# Get top-3 predicted class indices
test_pred = selecting_top_3(ensemble_pred)
test_pred = test_pred.astype('int32')

# Decode predictions into fertilizer names
test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)

print("Top-3 class indices for first 5 test rows:")
print(test_pred[:5])

print("\nðŸŒ±Top-3 predicted fertilizer names (first 5 rows):")
print(top_3_predictions[:5])


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(row) for row in top_3_predictions]
submission.to_csv('9_subs.csv')
del submission, xgb_test_preds, lgb_test_preds
gc.collect()


cat_features = [X.columns.get_loc(col) for col in feature_encoders.keys()]

# Step 1: Define base model (same as before)
lgb_base = LGBMClassifier(
    objective='multiclass',
    metric='multi_logloss',
    random_state=42,
    n_jobs=-1
)

# Step 2: Shrunk hyperparameter search space (faster tuning)
param_grid = {
    'num_leaves': [15, 31],
    'max_depth': [3, 4],
    'min_child_samples': [20, 50],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'learning_rate': [0.05],
    'n_estimators': [300]
}

# Step 3: Smaller Stratified K-Fold for faster tuning
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: RandomizedSearchCV
lgb_random_search = RandomizedSearchCV(
    estimator=lgb_base,
    param_distributions=param_grid,
    n_iter=8,  # just 8 combinations
    cv=skf,
    scoring='neg_log_loss',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Step 5: Run the hyperparameter search
lgb_random_search.fit(X, y_encoded, categorical_feature=cat_features)

# Step 6: Print best hyperparameters
print("âœ… BEST PARAMETERS FOUND:")
print(lgb_random_search.best_params_)

print("\nâœ… BEST CV LOG LOSS:")
print(-lgb_random_search.best_score_)


# LightGBM parameters after tuning
lgb_params = {
    'n_estimators': 300,
    'max_depth': 4,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_jobs': -1
}

# Stratified K-Fold
scores, lgb_test_preds = [], []

cat_features = [X.columns.get_loc(col) for col in feature_encoders.keys()]

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    lgb_md = LGBMClassifier(**lgb_params)
    lgb_md.set_params(early_stopping_rounds=50, verbose=-1)

    lgb_md.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               eval_metric='multi_logloss',
               categorical_feature=cat_features)

    y_proba = lgb_md.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    lgb_test_preds.append(lgb_md.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

lgb_cv_mean = np.mean(scores)
lgb_cv_sd = np.std(scores)
print(f"\nLightGBM CV MAP@3: {lgb_cv_mean:.4f} Â± {lgb_cv_sd:.4f}")


pred_agg = 0
for i in range(len(lgb_test_preds)):
    pred_agg += lgb_test_preds[i] / len(lgb_test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('10_sub.csv')

del submission, lgb_test_preds
gc.collect()


# Interaction feature
X['Crop_Soil'] = X['Crop Type'].astype(str) + "_" + X['Soil Type'].astype(str)
test['Crop_Soil'] = test['Crop Type'].astype(str) + "_" + test['Soil Type'].astype(str)

le_crop_soil = LabelEncoder()
X['Crop_Soil'] = le_crop_soil.fit_transform(X['Crop_Soil'])
test['Crop_Soil'] = le_crop_soil.transform(test['Crop_Soil'])

# Categorical features list
cat_features = [X.columns.get_loc(col) for col in ['Soil Type', 'Crop Type', 'Crop_Soil']]


# Best LightGBM hyperparameters
lgb_params = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_jobs': -1
}

# LightGBM categorical feature indices
cat_features = [X.columns.get_loc(col) for col in ['Soil Type', 'Crop Type']]

# Train and evaluate model
scores, lgb_test_preds = [], []

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    lgb_md = LGBMClassifier(**lgb_params)

    lgb_md.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        categorical_feature=cat_features,
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=0)
        ]
    )

    y_proba = lgb_md.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.5f}")
    scores.append(score)

    lgb_test_preds.append(lgb_md.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

# Summary of performance
lgb_cv_mean = np.mean(scores)
lgb_cv_sd = np.std(scores)
print(f"\nâœ… LightGBM CV MAP@3: {lgb_cv_mean:.5f} Â± {lgb_cv_sd:.5f}")


pred_agg = 0
for i in range(len(lgb_test_preds)):
    pred_agg += lgb_test_preds[i] / len(lgb_test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('16_sub.csv')

del submission, lgb_test_preds
gc.collect()


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'random_state': 42,
        'n_jobs': -1
    }

    scores = []

    for train_ix, val_ix in skf.split(X, y_encoded):
        X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
        y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            categorical_feature=cat_features,
            callbacks=[
                early_stopping(50, verbose=False),
                log_evaluation(0)
            ]
        )

        y_proba = model.predict_proba(X_val)
        top_3_preds = selecting_top_3(y_proba)
        score = mapk(y_val, top_3_preds, k=3)
        scores.append(score)

        del X_train, X_val, y_train, y_val
        gc.collect()

    return np.mean(scores)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

print("BEST PARAMETERS FOUND:")
print(study.best_params)

print("\nBEST CV MAP@3:")
print(study.best_value)


lgb_params = {
    'n_estimators': 514,
    'learning_rate': 0.08060637943440198,
    'max_depth': 6,
    'num_leaves': 24,
    'min_child_samples': 37,
    'subsample': 0.7726863931666217,
    'colsample_bytree': 0.6752859308230853,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_jobs': -1
}

scores, lgb_test_preds = [], []

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    model = LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        categorical_feature=cat_features,
        callbacks=[
            early_stopping(50, verbose=False),
            log_evaluation(0)
        ]
    )

    y_proba = model.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    lgb_test_preds.append(model.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

# Print final CV score
lgb_cv_mean = np.mean(scores)
lgb_cv_sd = np.std(scores)
print(f"\n LightGBM Optuna CV MAP@3: {lgb_cv_mean:.5f} Â± {lgb_cv_sd:.5f}")


pred_agg = 0
for i in range(len(lgb_test_preds)):
    pred_agg += lgb_test_preds[i] / len(lgb_test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('21_sub.csv')

del submission, lgb_test_preds
gc.collect()


# Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 40),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'n_jobs': -1,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_ix, val_ix in skf.split(X, y_encoded):
        X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
        y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_proba = model.predict_proba(X_val)
        top_3_preds = selecting_top_3(y_proba)
        score = mapk(y_val, top_3_preds, k=3)
        scores.append(score)

        del X_train, X_val, y_train, y_val
        gc.collect()

    return np.mean(scores)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)  # Reduce this further if needed

# Print results
print("BEST XGBOOST PARAMETERS:")
print(study.best_trial.params)

print("\nBEST CV MAP@3:")
print(study.best_trial.value)


# Best hyperparameters from Optuna
xgb_best_params = {
    'n_estimators': 355,
    'learning_rate': 0.0884853159251951,
    'max_depth': 5,
    'min_child_weight': 24,
    'subsample': 0.9283123309515314,
    'colsample_bytree': 0.9134589472537094,
    'gamma': 0.025308910820470265,
    'n_jobs': -1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
}

# Reinitialize for retraining
xgb_test_preds = []
scores = []

for i, (train_ix, val_ix) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X.iloc[train_ix], X.iloc[val_ix]
    y_train, y_val = y_encoded[train_ix], y_encoded[val_ix]

    xgb_md = XGBClassifier(**xgb_best_params)
    xgb_md.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_proba = xgb_md.predict_proba(X_val)
    top_3_preds = selecting_top_3(y_proba)
    score = mapk(y_val, top_3_preds, k=3)

    print(f"Fold {i+1} MAP@3: {score:.4f}")
    scores.append(score)

    xgb_test_preds.append(xgb_md.predict_proba(test))

    del X_train, X_val, y_train, y_val
    gc.collect()

xgb_cv_mean = np.mean(scores)
xgb_cv_sd = np.std(scores)
print(f"\nFinal XGBoost CV MAP@3: {xgb_cv_mean:.5f} Â± {xgb_cv_sd:.5f}")


pred_agg = 0
for i in range(len(xgb_test_preds)):
    pred_agg += xgb_test_preds[i] / len(xgb_test_preds)

test_pred = selecting_top_3(pred_agg)
test_pred = test_pred.astype('int32')

test_shape = test_pred.shape
top_3_predictions = le_target.inverse_transform(test_pred.reshape(-1, 1))
top_3_predictions = top_3_predictions.reshape(test_shape)

submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(each) for each in top_3_predictions]
submission.head(10)


submission.to_csv('32_sub.csv')

del submission, xgb_test_preds
gc.collect()

