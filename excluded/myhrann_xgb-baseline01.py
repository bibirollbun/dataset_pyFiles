import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score



train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


X = train_df.drop(columns=["id", "rainfall"])
y = train_df["rainfall"]


weights = class_weight.compute_sample_weight(class_weight="balanced", y=y)


X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
    X, y, weights, test_size=0.2, random_state=42, stratify=y
)


model = XGBRegressor(objective="binary:logistic", eval_metric="logloss", random_state=42)
model.fit(X_train, y_train, sample_weight=weights_train)


y_val_pred = model.predict(X_val)
auc_score = roc_auc_score(y_val, y_val_pred)
print(f"AUC Score: {auc_score}")


X_test = test_df.drop(columns=["id"])
test_preds = model.predict(X_test)


submission = sample_submission_df.copy()
submission["rainfall"] = test_preds
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")




