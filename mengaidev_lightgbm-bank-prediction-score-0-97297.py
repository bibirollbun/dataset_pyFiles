import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
import os

# Input data files are available in the read-only "../input/" directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


categorical_cols = [col for col in train.columns if train[col].dtype == "object"]


for col in categorical_cols:
    print(f"{col} , {train[col].nunique()}")


numerical_cols = [col for col in train.columns if (train[col].dtype in ["int64", "float64"]) and col not in ["id", "y"]]


numerical_cols


train[numerical_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()


categorical_cols = [col for col in train.columns if train[col].dtype == "object"]


for feature in categorical_cols:
    train[feature] = train[feature].astype("category")
    test[feature] = test[feature].astype("category")


X = train.drop(columns=['y'])
y = train['y']
X_test = test


combined_df = pd.concat([X.drop('id', axis=1), X_test.drop('id', axis=1)], axis=0)
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)
X = combined_df.iloc[:len(X)]
X_test = combined_df.iloc[len(X):]


n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

y_proba_val_rf = np.zeros(len(X))
y_probs_rf = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nTraining fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_proba_val_rf[val_idx] = rf.predict_proba(X_val)[:, 1]
    y_probs_rf += rf.predict_proba(X_test)[:, 1] / n_splits


n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

y_probs_val = np.zeros(len(X))
y_probs = np.zeros(len(X_test))

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
        device='cuda'
    )

    xgb_clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )

    y_probs_val[val_idx] = xgb_clf.predict_proba(X_val)[:, 1]
    y_probs += xgb_clf.predict_proba(X_test)[:, 1] / n_splits


n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

y_probs_lgb = np.zeros(len(X_test))
y_probs_val_lgb = np.zeros(len(X))

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
        verbosity=-1
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

    y_probs_val_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
    y_probs_lgb += model.predict_proba(X_test)[:, 1] / n_splits


class Config:
    state = 42
    early_stop = 100

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.state)

y_probs_val_cat = np.zeros(len(X))
y_probs_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    test_pool = Pool(X_test)

    cat_clf = CatBoostClassifier(
        random_state=Config.state,
        early_stopping_rounds=Config.early_stop,
        eval_metric="Logloss",
        n_estimators=5000,
        learning_rate=0.06524873965257823,
        l2_leaf_reg=0.8867612905712001,
        bagging_temperature=0.1317347791955057,
        random_strength=0.9922857768340815,
        depth=7,
        min_data_in_leaf=8,
        task_type="CPU",
        verbose=100
    )

    cat_clf.fit(train_pool, eval_set=val_pool, use_best_model=True)
    y_probs_val_cat[val_idx] = cat_clf.predict_proba(val_pool)[:, 1]
    y_probs_cat += cat_clf.predict_proba(test_pool)[:, 1] / n_splits


def plot_feature_importance(model, X, title="Feature Importance"):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances.sort_values().plot(kind='barh')
        plt.title(title)
        plt.show()

plot_feature_importance(xgb_clf, X, "XGBoost Feature Importance")
plot_feature_importance(model, X, "LightGBM Feature Importance")
plot_feature_importance(cat_clf, X, "CatBoost Feature Importance")


oof_auc_xgb = roc_auc_score(y, y_probs_val)
oof_auc_lgb = roc_auc_score(y, y_probs_val_lgb)
oof_auc_cat = roc_auc_score(y, y_probs_val_cat)
oof_auc_rf = roc_auc_score(y, y_proba_val_rf)
print(oof_auc_rf)
print(f"XGBoost OOF AUC: {oof_auc_xgb:.5f}")
print(f"LightGBM OOF AUC: {oof_auc_lgb:.5f}")
print(f"CatBoost OOF AUC: {oof_auc_cat:.5f}")


from scipy.optimize import minimize

def find_best_weights(oof_preds, true_labels):
    def neg_roc_auc(weights):
        weighted_preds = np.sum(oof_preds * weights, axis=1)
        return -roc_auc_score(true_labels, weighted_preds)

    initial_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
    bounds = [(0, 1)] * oof_preds.shape[1]
    
    result = minimize(neg_roc_auc, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    print(f"Best Blending Score: {-result.fun:.5f}")
    return result.x

oof_predictions = np.column_stack([
    y_probs_val,
    y_probs_val_lgb,
    y_probs_val_cat,
    y_proba_val_rf
])

best_weights = find_best_weights(oof_predictions, y)
print(f"Best Weights (XGB, LGB, CAT, RF): {best_weights}")


from lightgbm import LGBMClassifier

meta_model = LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

meta_model.fit(oof_predictions, y)
meta_oof_preds = meta_model.predict_proba(oof_predictions)[:, 1]
auc_score = roc_auc_score(y, meta_oof_preds)
print(f"Meta model OOF AUC: {auc_score:.5f}")


oof_predictions_test = np.column_stack([
    y_probs,
    y_probs_lgb,
    y_probs_cat,
    y_probs_rf
])
pred = meta_model.predict_proba(oof_predictions_test)[:,1]


submission = pd.DataFrame({
    "id": test["id"],
    "y": pred
})
submission = submission.sort_values(by="y", ascending=True)
submission.to_csv("submission.csv", index=False)


submission

