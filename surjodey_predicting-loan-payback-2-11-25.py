import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import numpy as np


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


y = train['loan_paid_back']
X = train.drop(['loan_paid_back'], axis=1)


for col in X.select_dtypes(include='object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Validation split (for local score)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.018,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    random_state=42
)
model.fit(X_train, y_train)


val_preds = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print("Validation ROC-AUC:", round(val_auc, 5))
# Predict
test_preds = model.predict_proba(test)[:, 1]


sorted_preds = np.sort(test_preds)[::-1]  # high prob first
sorted_preds = np.clip(sorted_preds, 0.28, 0.72)

submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': sorted_preds
})
submission.to_csv('submission_lgb.csv', index=False)
print("Submission saved → submission_lgb.csv")

