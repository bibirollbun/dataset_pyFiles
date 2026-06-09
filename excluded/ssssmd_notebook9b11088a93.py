import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt


base_path = "/kaggle/input/otto-group-product-classification-challenge/"
train_df = pd.read_csv(base_path + "train.csv")
test_df = pd.read_csv(base_path + "test.csv")
sample_submission = pd.read_csv(base_path + "sampleSubmission.csv")

X = train_df.drop(['id', 'target'], axis=1)
y = train_df['target'].str.replace("Class_", "").astype(int) - 1
X_test = test_df.drop('id', axis=1)


model_tmp = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0
)
model_tmp.fit(X, y)
importance_scores = model_tmp.feature_importances_

feat_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importance_scores
}).sort_values(by='importance', ascending=False)

top_feats = feat_importance_df['feature'].head(20).tolist()
print(top_feats)


def generate_feature_extensions(X_base, top_feats):
    X_ext_list = []
    for f in top_feats:
        X_ext_list.append(pd.DataFrame({
            f + '_log': np.log1p(X_base[f]),
            f + '_sqr': X_base[f] ** 2,
            f + '_sqrt': np.sqrt(X_base[f])
        }))
    X_ext = pd.concat(X_ext_list, axis=1)
    return X_ext

X_ext = generate_feature_extensions(X, top_feats)
X_test_ext = generate_feature_extensions(X_test, top_feats)


X_full = pd.concat([X.reset_index(drop=True), X_ext.reset_index(drop=True)], axis=1)
X_test_full = pd.concat([X_test.reset_index(drop=True), X_test_ext.reset_index(drop=True)], axis=1)

X_full = X_full.fillna(X_full.median())
X_test_full = X_test_full.fillna(X_full.median())


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros((X_full.shape[0], 9))
test_preds = np.zeros((X_test_full.shape[0], 9))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y)):
    print(f"Fold {fold+1}")
    X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=fold,
        early_stopping_rounds=200
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    oof_preds[val_idx] = model.predict_proba(X_val)
    test_preds += model.predict_proba(X_test_full) / n_splits


cv_score = log_loss(y, oof_preds)
print(f"CV LogLoss: {cv_score:.5f}")

submission = pd.DataFrame(test_preds, columns=sample_submission.columns[1:])
submission.insert(0, "id", test_df["id"])
submission.to_csv("submission.csv", index=False)

