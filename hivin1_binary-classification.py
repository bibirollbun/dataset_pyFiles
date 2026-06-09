import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

test_ids = test['id']

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# quick look
print(train.head())


print(train.info())
print(train.describe())


print(train['y'].value_counts())
sns.countplot(x='y', data=train)
plt.title("y Distribution")
plt.show()


print(train.isnull().sum())


X = train.drop('y', axis=1)
y = train['y']
# one-hot encoding for categoricals
X = pd.get_dummies(X)
test = pd.get_dummies(test)
# align columns 
X, test = X.align(test, join='left', axis=1, fill_value=0)
# split train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test = scaler.transform(test)


#Model Training
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_val)
print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))


# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
print("Random Forest Accuracy:", accuracy_score(y_val, y_pred_rf))


print("Classification Report (Random Forest):")
print(classification_report(y_val, y_pred_rf))


cm = confusion_matrix(y_val, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# Final Submission
final_preds = rf.predict(test)
submission = pd.DataFrame({
    'id': test_ids,       # use the saved ids
    'target': final_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved with correct id column!")






