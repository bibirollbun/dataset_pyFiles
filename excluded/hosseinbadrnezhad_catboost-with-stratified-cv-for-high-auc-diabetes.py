!pip install catboost


!pip install shap


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt


DATA_DIR = '/kaggle/input/playground-series-s5e12'
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

y = train["diagnosed_diabetes"]
X = train.drop(columns=["diagnosed_diabetes"])


cat_features = X.select_dtypes(include=["object"]).columns.tolist()
print("Categorical:", cat_features)


for c in X.columns:
    if c in cat_features:
        X[c] = X[c].fillna("Unknown")
        test[c] = test[c].fillna("Unknown")
    else:
        X[c] = X[c].fillna(X[c].median())
        test[c] = test[c].fillna(X[c].median())


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_pred = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    valid_pool = Pool(X_va, y_va, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        l2_leaf_reg=5,
        bagging_temperature=0.3,
        random_strength=1.5,
        od_wait=80,
        task_type="CPU",
        verbose=False
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    oof[va_idx] = model.predict_proba(X_va)[:, 1]
    test_pred += model.predict_proba(test)[:, 1] / skf.n_splits

print("OOF AUC:", roc_auc_score(y, oof))


final_pool = Pool(X, y, cat_features=cat_features)

final_model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.02,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    l2_leaf_reg=5,
    bagging_temperature=0.3,
    random_strength=1.5,
    task_type="CPU",
    verbose=False
)

final_model.fit(final_pool)



explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)


shap.summary_plot(shap_values, X)


shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    max_display=15
)


from sklearn.metrics import confusion_matrix

pred_label = (oof > 0.5).astype(int)
fn_idx = (y == 1) & (pred_label == 0)

shap.summary_plot(
    shap_values[fn_idx],
    X.loc[fn_idx]
)


mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_feature = X.columns[np.argmax(mean_abs_shap)]

shap.dependence_plot(
    top_feature,
    shap_values,
    X
)


sub["diagnosed_diabetes"] = test_pred
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")

