import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
train  = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test   = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')



print(train.shape)
print(test.shape)
print(sample.shape)
print(train.columns)



TARGET = "diagnosed_diabetes"

X = train.drop(columns=["id", TARGET])
y = train[TARGET]

X_test = test.drop(columns=["id"])



all_data = pd.concat([X, X_test], axis=0)
all_data_encoded = pd.get_dummies(all_data, drop_first=True)

X_encoded = all_data_encoded.iloc[:len(X)]
X_test_encoded = all_data_encoded.iloc[len(X):]



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_encoded)
X_test_scaled = scaler.transform(X_test_encoded)



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []

for train_idx, val_idx in skf.split(X_scaled, y):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    auc_scores.append(roc_auc_score(y_val, preds))

print("Mean CV AUC:", np.mean(auc_scores))



final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_scaled, y)



test_preds = final_model.predict_proba(X_test_scaled)[:, 1]



submission = sample.copy()
submission["diagnosed_diabetes"] = test_preds

submission.to_csv("submission.csv", index=False)
submission.head()



print(submission.shape)
print(submission.columns)
print(submission.head())
print(
    submission["diagnosed_diabetes"].min(),
    submission["diagnosed_diabetes"].max()
)





