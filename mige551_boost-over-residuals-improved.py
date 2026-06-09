import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge




train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")



target_col = "loan_paid_back"

train_features = train.drop(columns=[target_col])
test_features = test.copy()

full = pd.concat([train_features, test_features], axis=0, ignore_index=True)

cat_cols = full.select_dtypes(include=["object"]).columns
cat_cols



full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=False)

n_train = len(train)
full_train = full_encoded.iloc[:n_train, :]
full_test  = full_encoded.iloc[n_train:, :]

X = full_train.drop(columns=["id"])
X_test = full_test.drop(columns=["id"])

y = train[target_col].values



model1 = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)

model1.fit(X, y)

pred0 = model1.predict_proba(X)[:, 1]
print("Stage1 mean pred:", pred0.mean())



new_target = pred0 - y



model2 = Ridge(alpha=1.0)  # 過学習しにくい
model2.fit(X, new_target)



# stage1
test_pred0 = model1.predict_proba(X_test)[:, 1]

# stage2
correction = model2.predict(X_test)

# combine
raw = test_pred0 + correction

# sigmoid
final_pred = 1 / (1 + np.exp(-raw))



submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": final_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()


