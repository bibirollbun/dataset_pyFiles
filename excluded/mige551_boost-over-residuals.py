import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, LGBMRegressor



train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")



target_col = "loan_paid_back"

train_features = train.drop(columns=[target_col])
test_features = test.copy()

full = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# detect categoricals
cat_cols = full.select_dtypes(include=["object"]).columns
cat_cols



full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=False)

n_train = len(train)
full_train = full_encoded.iloc[:n_train, :]
full_test  = full_encoded.iloc[n_train:, :]

X = full_train.drop(columns=["id"])
X_test = full_test.drop(columns=["id"])

y = train[target_col]



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model1 = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model1.fit(X_train, y_train)

pred_val = model1.predict_proba(X_val)[:, 1]
print("Stage1 AUC:", roc_auc_score(y_val, pred_val))



def to_logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1-p))

# Stage1 predictions on full training data
pred0 = model1.predict_proba(X)[:, 1]
pred0_logit = to_logit(pred0)

true_logit = to_logit(y.values)

residual = true_logit - pred0_logit

X2 = X.copy()
y2 = residual



model2 = LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

model2.fit(X2, y2)



# stage1
test_pred0 = model1.predict_proba(X_test)[:, 1]
test_pred0_logit = to_logit(test_pred0)

# stage2
residual_test = model2.predict(X_test)

# combine
final_logit = test_pred0_logit + residual_test
final_pred = 1 / (1 + np.exp(-final_logit))



submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": final_pred
})

submission.to_csv("submission.csv", index=False)

submission.head()


