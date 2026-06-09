import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/input/tabular-playground-series-2025-diabetes"

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X = train.drop(columns=["id", "diagnosed_diabetes"])
y = train["diagnosed_diabetes"]
X_test = test.drop(columns=["id"])

cat_features = X.select_dtypes(include="object").columns.tolist()



from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200
)

model.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    early_stopping_rounds=100
)

print("CatBoost AUC:", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))



test_probs = model.predict_proba(X_test)[:, 1]

import pandas as pd
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_probs
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
submission.head()



len(test_probs), test_probs.min(), test_probs.max()


