import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

SEED = 42



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print(train.shape, test.shape)
train.head()


TARGET = "diagnosed_diabetes"
ID_COL = "id"

X = train.drop(columns=[TARGET, ID_COL])
y = train[TARGET]

X_test = test.drop(columns=[ID_COL])



num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns

print("Numeric:", len(num_features))
print("Categorical:", len(cat_features))



num_imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X[num_features] = num_imputer.fit_transform(X[num_features])
X[num_features] = scaler.fit_transform(X[num_features])

X_test[num_features] = num_imputer.transform(X_test[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])



from sklearn.preprocessing import OrdinalEncoder

cat_imputer = SimpleImputer(strategy="most_frequent")
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X[cat_features] = cat_imputer.fit_transform(X[cat_features])
X[cat_features] = encoder.fit_transform(X[cat_features])

X_test[cat_features] = cat_imputer.transform(X_test[cat_features])
X_test[cat_features] = encoder.transform(X_test[cat_features])



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)



lgb_model = LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=SEED
)



xgb_model = XGBClassifier(
    n_estimators=2500,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=SEED,
    tree_method="hist"
)



cat_model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.02,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    random_seed=SEED
)



models = {
    "lgb": lgb_model,
    "xgb": xgb_model,
    "cat": cat_model
}

oof_preds = {}
test_preds = {}

for name, model in models.items():
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits

    score = roc_auc_score(y, oof)
    print(f"{name.upper()} ROC-AUC: {score:.5f}")

    oof_preds[name] = oof
    test_preds[name] = test_pred



ensemble_oof = (
    0.4 * oof_preds["lgb"] +
    0.3 * oof_preds["xgb"] +
    0.3 * oof_preds["cat"]
)

ensemble_auc = roc_auc_score(y, ensemble_oof)
print("Ensemble ROC-AUC:", ensemble_auc)



final_test_pred = (
    0.4 * test_preds["lgb"] +
    0.3 * test_preds["xgb"] +
    0.3 * test_preds["cat"]
)



submission = pd.DataFrame({
    "id": test[ID_COL],
    "diagnosed_diabetes": final_test_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()





