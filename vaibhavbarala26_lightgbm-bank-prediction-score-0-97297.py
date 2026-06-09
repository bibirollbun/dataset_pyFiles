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



categorical_cols = [col for col in train.columns if train[col].dtype == "object"]


for col in categorical_cols:
    print(f"{col} , {train[col].nunique()}")


numerical_cols = [
    col for col in train.columns 
    if (train[col].dtype in ["int64", "float64"]) and col not in ["id", "y"]
]



numerical_cols


import seaborn as sns
import matplotlib.pyplot as plt
train[numerical_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
train[numerical_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()



class_counts = train['y'].value_counts()
print(class_counts)
scale_pos_weight = class_counts[0] / class_counts[1]  # majority / minority
print(f"scale_pos_weight: {scale_pos_weight:.2f}")



categorical_cols = [col for col in train.columns if train[col].dtype == "object"]


categorical_cols


for feature in categorical_cols:
        train[feature] = train[feature].astype("category")
for feature in categorical_cols:
        test[feature] = test[feature].astype("category")


X_test = test


X = train.drop(columns=['y'])
y = train['y']



combined_df = pd.concat([X.drop('id', axis=1), X_test.drop('id', axis=1)], axis=0)

combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

X = combined_df.iloc[:len(X)]
X_test = combined_df.iloc[len(X):]



# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# n_splits = 5
# kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# y_proba_val_rf = np.zeros(len(X))       # OOF predictions for validation
# y_probs_rf = np.zeros(len(X_test))   # averaged test predictions

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     print(f"\nTraining fold {fold + 1}/{n_splits} >>>")

#     X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#     X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

#     rf = RandomForestClassifier(
#         n_estimators=1000,
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         random_state=42,
#         n_jobs=-1
#     )

#     rf.fit(X_train, y_train)

#     # Validation predictions
#     y_pred = rf.predict(X_val)
#     y_proba_val_rf[val_idx] = rf.predict_proba(X_val)[:, 1]

#     # Test predictions (average across folds)
#     y_probs_rf += rf.predict_proba(X_test)[:, 1] / n_splits

#     # Metrics
#     print("Accuracy:", accuracy_score(y_val, y_pred))
#     print("ROC AUC:", roc_auc_score(y_val, y_proba_val_rf[val_idx]))
#     print("\nClassification Report:\n", classification_report(y_val, y_pred))

# # Overall OOF ROC AUC
# print(f"\nOverall OOF ROC AUC: {roc_auc_score(y, y_proba_val_rf):.5f}")



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

y_probs_val = np.zeros(len(X))   # OOF validation predictions for entire dataset
y_probs = np.zeros(len(X_test))  # averaged test set predictions

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nTraining fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    xgb_clf = XGBClassifier(
        tree_method='hist',
        n_estimators=10000,
        objective='binary:logistic',
        random_state=42,
        enable_categorical=True,
        eval_metric='auc',
        booster='gbtree',
        n_jobs=-1,
        reg_lambda=4.510522889747622,
        reg_alpha=5.007953193043952, 
        colsample_bytree=0.5831655543160346,
        subsample=0.9808690492838653,
        learning_rate=0.008247101477015132,
        max_depth=11,
        min_child_weight=1,
        device='cuda',
    )

    xgb_clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )

    # Store fold validation predictions in the correct place
    y_probs_val[val_idx] = xgb_clf.predict_proba(X_val)[:, 1]

    # Accumulate test set predictions, averaged over folds
    y_probs += xgb_clf.predict_proba(X_test)[:, 1] / n_splits



import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

y_probs_lgb = np.zeros(len(X_test))          # For averaged test set predictions
y_probs_val_lgb = np.zeros(len(X))           # For out-of-fold validation predictions

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.06,
        num_leaves=100,
        max_depth=10,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.79,
        reg_lambda=3.0,
        max_bin=4523,
        random_state=42,
        verbosity=-1,

    )
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=100)
        ]
    )

    # Save validation fold predictions into the correct positions of full OOF array
    y_probs_val_lgb[val_idx] = model.predict_proba(X_val)[:, 1]

    # Accumulate test predictions averaged over folds
    y_probs_lgb += model.predict_proba(X_test)[:, 1] / n_splits



X = train.drop(columns=['y'])
y = train['y']


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

# Config class with fixed values
class Config:
    state = 42
    early_stop = 100

# Assume categorical_cols is defined, e.g.
# categorical_cols = ['cat_col1', 'cat_col2', ...]

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.state)
y_probs_val_cat = np.zeros(len(X))
y_probs_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits} >>>")
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    val_pool = Pool(X_val, y_val, cat_features=categorical_cols)
    test_pool = Pool(X_test, cat_features=categorical_cols)

    cat_clf = CatBoostClassifier(
        random_state=Config.state,
        early_stopping_rounds=Config.early_stop,
        eval_metric="Logloss",
        n_estimators=10000,
        learning_rate=0.05524873965257823,
        l2_leaf_reg=0.8867612905712001,
        bagging_temperature=0.1317347791955057,
        random_strength=0.9922857768340815,
        depth=7,
        min_data_in_leaf=8,
        task_type="GPU",
        verbose=100
    )

    cat_clf.fit(train_pool, eval_set=val_pool, use_best_model=True)

    y_probs_val_cat[val_idx] = cat_clf.predict_proba(val_pool)[:, 1]
    y_probs_cat += cat_clf.predict_proba(test_pool)[:, 1] / n_splits



if hasattr(model, "feature_importances_"):
    importances = pd.Series(xgb_clf.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind='barh')



if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind='barh')



if hasattr(model, "feature_importances_"):
    importances = pd.Series(cat_clf.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind='barh')



from sklearn.metrics import roc_auc_score
oof_auc_xgb = roc_auc_score(y, y_probs_val)
oof_auc_lgb = roc_auc_score(y, y_probs_val_lgb)
oof_auc_cat = roc_auc_score(y, y_probs_val_cat)
print(f"XGBoost OOF AUC: {oof_auc_xgb:.5f}")
print(f"LightGBM OOF AUC: {oof_auc_lgb:.5f}")
print(f"CatBoost OOF AUC: {oof_auc_cat:.5f}")


from scipy.optimize import minimize

# --- Function to find the best weights ---
def find_best_weights(oof_preds, true_labels):
    """
    Finds the optimal weights for blending OOF predictions.
    """
    # Objective function to minimize (negative ROC AUC)
    def neg_roc_auc(weights):
        # Combine predictions with weights
        weighted_preds = np.sum(oof_preds * weights, axis=1)
        # We want to MAXIMIZE AUC, so we MINIMIZE negative AUC
        return -roc_auc_score(true_labels, weighted_preds)

    # Initial guess for weights (equal weighting)
    initial_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]
    
    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
    
    # Bounds: each weight must be between 0 and 1
    bounds = [(0, 1)] * oof_preds.shape[1]
    
    # Perform the optimization
    result = minimize(neg_roc_auc, 
                      initial_weights, 
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints)
    
    print(f"Best Blending Score: {-result.fun:.5f}")
    return result.x

# Stack the OOF predictions column-wise
oof_predictions = np.column_stack([
    y_probs_val,
    y_probs_val_lgb,
    y_probs_val_cat,
])

# Find the best weights
best_weights = find_best_weights(oof_predictions, y)

print(f"Best Weights (XGB, LGB, CAT): {best_weights}")


oof_predictions = np.column_stack([
    y_probs_val,
    y_probs_val_lgb,
    y_probs_val_cat,
])


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# oof_predictions: numpy array shape (num_samples, num_base_models)
# y: true labels (array or Series)

# Initialize LightGBM classifier as meta-model
meta_model = LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# Train meta-model on OOF predictions
meta_model.fit(oof_predictions, y)

# Predict probabilities on OOF predictions to evaluate
meta_oof_preds = meta_model.predict_proba(oof_predictions)[:, 1]
auc_score = roc_auc_score(y, meta_oof_preds)

print(f"Meta model OOF AUC: {auc_score:.5f}")



oof_predictions = np.column_stack([
    y_probs,
    y_probs_lgb,
    y_probs_cat,
])
pred = meta_model.predict_proba(oof_predictions)[:,1]
pred 


pred = meta_model.predict_proba(oof_predictions)[:,1]
pred 


s = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
s


submission = pd.DataFrame({
    "id": test["id"],
    "y": pred
})

# Sort by 'y' in ascending order (use ascending=False for descending)
submission = submission.sort_values(by="y", ascending=True)

submission.to_csv("submission.csv", index=False)



submission




