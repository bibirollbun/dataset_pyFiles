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
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, confusion_matrix, classification_report



train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")



train.head()


train.info()


categorical_cols = [col for col in train.columns if train[col].dtype == "object"]


# Convert object columns to category dtype
for col in categorical_cols:
    train[col] = train[col].astype('category')
for col in categorical_cols:
    test[col] = test[col].astype('category')




for col in categorical_cols:
    print(f"{col} , {train[col].nunique()}")


categorical_cols


train.isna().sum()


# test['balance'] = np.log1p(test['balance'].clip(lower=0))
# test['duration'] = np.log1p(test['duration'].clip(lower=0))
# train['balance'] = np.log1p(train['balance'].clip(lower=0))
# train['duration'] = np.log1p(train['duration'].clip(lower=0))
# train['balance'] = np.log1p(train['balance'].clip(lower=0))
# train['duration'] = np.log1p(train['duration'].clip(lower=0))

# train['balance'] = np.log1p(train['balance']+0.00000001)
# train['duration'] = np.log1p(train['duration']+0.0000001)

# test['balance'] = np.log1p(test['balance']+0.00000001)
# test['duration'] = np.log1p(test['duration']+0.0000001)
# for col in ['campaign', 'previous', 'pdays']:
#     upper_limit = test[col].quantile(0.99)
#     test[col] = np.clip(test[col], a_min=None, a_max=upper_limit)
# for col in ['campaign', 'previous', 'pdays']:
#     upper_limit = train[col].quantile(0.99)
#     train[col] = np.clip(train[col], a_min=None, a_max=upper_limit)



import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=train, x='y')
plt.title("Target Variable Distribution (y)")
plt.show()



numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
train[numeric_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()




for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=train, x=col, hue='y', order=train[col].value_counts().index)
    plt.title(f"{col} vs Target (y)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



plt.figure(figsize=(10, 8))
corr = train[numeric_cols + ['y']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()



class_counts = train['y'].value_counts()
print(class_counts)
scale_pos_weight = class_counts[0] / class_counts[1]  # majority / minority
print(f"scale_pos_weight: {scale_pos_weight:.2f}")



import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score



X = train.drop(columns=['y'])
y = train['y']


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define model
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    enable_categorical=True,
    learning_rate=0.03718332260189157,
    max_depth=7,
    subsample=0.8159016609332052,
    colsample_bytree=0.7199543680856613,
    n_estimators=10000,  # large enough for early stopping
    tree_method='gpu_hist',           # ğŸ‘ˆ Enable GPU
    predictor='gpu_predictor',        # ğŸ‘ˆ Optional, for GPU inference
    random_state=42
)


xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)




from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns



# Predict class labels and probabilities
y_val_pred_proba = xgb_clf.predict_proba(X_val)[:, 1]
y_val_pred = xgb_clf.predict(X_val)

# AUC
val_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"ğŸ”µ AUC-ROC: {val_auc:.4f}")

# Other metrics
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print(f"âœ… Accuracy:  {acc:.4f}")
print(f"ğŸ�¯ Precision: {prec:.4f}")
print(f"ğŸ”� Recall:    {rec:.4f}")
print(f"ğŸ“Š F1 Score:  {f1:.4f}")



print("\nğŸ“ƒ Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=["no", "yes"]))



fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', label=f"AUC = {val_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()



X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Stratify helps maintain class balance, good practice for classification
)
w_train = compute_sample_weight(class_weight='balanced', y=y_train)


# --- 4. Define and train the LightGBM lgb_clf ---
print("\nğŸ“¦ Training the LightGBM lgb_clf...")
lgb_clf = lgb.LGBMClassifier(
    max_depth=7,
    device='gpu',  # Use 'gpu' or remove if you want to use CPU
    n_estimators=10000,
    learning_rate=0.06,
    reg_alpha=0.8,
    reg_lambda=3.0,
    colsample_bytree=0.5,
    subsample=0.8,
    random_state=42,
    verbosity=-1
)

lgb_clf.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_val, y_val)],
    eval_metric='logloss',
    callbacks=[lgb.early_stopping(100, verbose=True)] # Early stopping will prevent overfitting
)


# --- 5. Evaluate the lgb_clf ---
print("\nEvaluating the lgb_clf...")
lgb_preds = lgb_clf.predict_proba(X_val, num_iteration=lgb_clf.best_iteration_)[:, 1]
val_loss = log_loss(y_val, lgb_preds)
print(f"âœ… Final validation log loss: {val_loss:.4f}")



# Predict class labels and probabilities
y_val_pred_proba = lgb_clf.predict_proba(X_val)[:, 1]
y_val_pred = lgb_clf.predict(X_val)

# AUC
val_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"ğŸ”µ AUC-ROC: {val_auc:.4f}")

# Other metrics
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print(f"âœ… Accuracy:  {acc:.4f}")
print(f"ğŸ�¯ Precision: {prec:.4f}")
print(f"ğŸ”� Recall:    {rec:.4f}")
print(f"ğŸ“Š F1 Score:  {f1:.4f}")



fpr, tpr, thresholds = roc_curve(y_val, lgb_preds)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', label=f"AUC = {val_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()



from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

params = {
    'iterations': 10000,
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 10,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 0.8,
    'random_strength': 0.5,
    'border_count': 254,
    'leaf_estimation_iterations': 10,
    'random_seed': 42,
    'verbose': 200,
    'early_stopping_rounds': 200,
    'eval_metric': 'AUC',
    'auto_class_weights': 'Balanced',
    'task_type': 'GPU'  # ğŸ‘ˆ Enable GPU
}


# Optional: define categorical features if any
# categorical_features = ['cat_col1', 'cat_col2', ...]

# Create Pool objects for train and validation
train_pool = Pool(data=X_train, label=y_train , cat_features=categorical_cols)  # , cat_features=categorical_features)
val_pool = Pool(data=X_val, label=y_val , cat_features=categorical_cols)        # , cat_features=categorical_features)

# Initialize and train cat_clf
cat_clf = CatBoostClassifier(**params)
cat_clf.fit(train_pool, eval_set=val_pool)

# Predict and evaluate
cat_pred = cat_clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, cat_pred)
print(f"Validation AUC: {auc:.5f}")



# Predict class labels and probabilities
y_val_pred_proba = cat_clf.predict_proba(X_val)[:, 1]
y_val_pred = cat_clf.predict(X_val)

# AUC
val_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"ğŸ”µ AUC-ROC: {val_auc:.4f}")

# Other metrics
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print(f"âœ… Accuracy:  {acc:.4f}")
print(f"ğŸ�¯ Precision: {prec:.4f}")
print(f"ğŸ”� Recall:    {rec:.4f}")
print(f"ğŸ“Š F1 Score:  {f1:.4f}")



fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', label=f"AUC = {val_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()



lgb_preds = lgb_clf.predict_proba(X_val)[:, 1]
xgb_preds = xgb_clf.predict_proba(X_val)[:, 1]
cat_preds = cat_clf.predict_proba(X_val)[:, 1]



# from sklearn.metrics import roc_auc_score
# import numpy as np

# best_auc = 0
# best_weights = (0, 0, 0)
# results = []

# # Search all combinations where w1 + w2 + w3 = 1
# for w1 in np.arange(0, 1.05, 0.1):
#     for w2 in np.arange(0, 1.05 - w1, 0.1):
#         w3 = 1.0 - w1 - w2
#         if w3 < 0 or w3 > 1:
#             continue

#         # Weighted average of predictions
#         ensemble_preds = w1 * lgb_preds + w2 * xgb_preds + w3 * cat_preds
#         auc = roc_auc_score(y_val, ensemble_preds)
#         results.append(((w1, w2, w3), auc))

#         if auc > best_auc:
#             best_auc = auc
#             best_weights = (w1, w2, w3)

# # Print best combination
# w1, w2, w3 = best_weights
# print(f"âœ… Best AUC: {best_auc:.5f} with weights -> LGB: {w1:.2f}, XGB: {w2:.2f}, CAT: {w3:.2f}")



xgb_preds = xgb_clf.fit(X,y)
lgb_preds = lgb_clf.fit(X,y)



train_pool = Pool(data=X, label=y , cat_features=categorical_cols)  # , cat_features=categorical_features)



cat_preds = cat_clf.fit(train_pool)


lgb_preds = lgb_clf.predict_proba(test)[:, 1]
xgb_preds = xgb_clf.predict_proba(test)[:, 1]
cat_preds = cat_clf.predict_proba(test)[:, 1]



# Weighted average ensemble
ensemble_preds = 0.30 * lgb_preds + 0.30 * xgb_preds + 0.40 * cat_preds



submission2 = pd.DataFrame({
    "id":test["id"],
    "y":ensemble_preds
})
submission2.to_csv("submission.csv",index=False)


# model.fit(X , y)
# xgb_clf.fit(X,y)


# lgbm_preds = model.predict_proba(test)[:, 1]
# xgb_preds = xgb_clf.predict_proba(test)[:, 1]


# ensemble_preds = 0.65 * lgbm_preds + 0.35 * xgb_preds


# submission2 = pd.DataFrame({
#     "id":test["id"],
#     "y":ensemble_preds
# })
# submission2.to_csv("submission.csv",index=False)





# import optuna
# import numpy as np
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# from sklearn.metrics import roc_auc_score

# # Train-validation split (keep stratification)
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Optional: check class imbalance ratio
# scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()



# def objective(trial):
#     params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'use_label_encoder': False,
#         'scale_pos_weight': scale_pos_weight,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'n_estimators': 1000,  # high so early stopping can cut it
#         'random_state': 42,
#         "enable_categorical":True,
#         "early_stopping_rounds":50,

#     }

#     model = XGBClassifier(**params)

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         verbose=False
#     )

#     y_val_pred = model.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, y_val_pred)
#     return auc



# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=30)

# print("ğŸ�¯ Best AUC:", study.best_value)
# print("âœ… Best Parameters:")
# print(study.best_params)


























