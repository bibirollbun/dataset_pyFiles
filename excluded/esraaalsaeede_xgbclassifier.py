import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.head(5)


print(train.isnull().sum())


train.info()


train.columns


train.shape


# Drop irrelevant columns
train.drop(columns=['id', 'day'], inplace=True)
test.drop(columns=['id', 'day'], inplace=True)


plt.figure(figsize=(10,6))
sns.heatmap(train.corr(), annot=True, cmap="coolwarm")
plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(data=train)
plt.xticks(rotation=90)
plt.show()


X = train.drop(columns=['rainfall']) 
y = train['rainfall']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)



X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)



model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='auc')
model.fit(X_train, y_train)


y_pred_prob = model.predict_proba(X_valid)[:, 1]  
print("Validation AUC Score:", roc_auc_score(y_valid, y_pred_prob))



test_predictions = model.predict_proba(test_scaled)[:, 1] 


submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

submission['rainfall'] = test_predictions

submission.to_csv("submission.csv", index=False)


submission.head()











