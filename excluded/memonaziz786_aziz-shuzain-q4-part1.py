
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("/kaggle/input/predict-who-is-more-influential-in-a-social-network/train.csv")
test = pd.read_csv("/kaggle/input/predict-who-is-more-influential-in-a-social-network/test.csv")

X = train.drop("Choice", axis=1)
y = train["Choice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Validation Accuracy:", acc)
test_pred = model.predict(test)
submission = pd.DataFrame({
    "Id": range(1, len(test_pred) + 1),
    "Choice": test_pred
})
print("Predictions are saved to submission_xgboost.csv")
submission.to_csv("submission_xgboost.csv", index=False)
 

