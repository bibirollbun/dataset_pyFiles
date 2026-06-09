import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


train_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


features = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']
train_df = train_df.dropna(subset=features)
test_df = test_df.fillna(0)

X = train_df[features]
y = train_df['isFraud']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df[features])


model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC: {auc_score:.4f}")


test_preds = model.predict_proba(X_test_scaled)[:, 1]


submission = pd.DataFrame({'TransactionID': test_df['TransactionID'], 'isFraud': test_preds})
print(submission)
submission.to_csv('baseline_submission.csv', index=False)




