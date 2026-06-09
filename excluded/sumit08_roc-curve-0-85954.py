import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import warnings


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.head()


train.isnull().sum()


test.head()


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mode()[0])


plt.figure(figsize=(12,6))
sn.scatterplot(x="id",y="winddirection", data= test)


test.isnull().sum()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


X = train.drop(columns=["id", "rainfall"])
y = train["rainfall"]
X_test_kaggle = test.drop(columns=["id"])


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test_kaggle = scaler.transform(X_test_kaggle)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


!pip install optuna


import optuna


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'random_state': 42
    }
    model = LGBMClassifier(**param)
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    return auc(fpr, tpr)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print('Best trial:', study.best_trial.params)


model = LGBMClassifier(**study.best_trial.params)
model.fit(X_train, y_train)


y_probs = model.predict_proba(X_val)[:, 1]


fpr, tpr, _ = roc_curve(y_val, y_probs)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
sn.set_style("whitegrid")
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()



print(f"AUC Score: {roc_auc:.4f}")


y_test_preds = model.predict_proba(X_test_kaggle)[:, 1]


import os

# Create the directory if it doesn't exist
os.makedirs("/mnt/data", exist_ok=True)

# Now save the file
sample["rainfall"] = y_test_preds
sample.to_csv("/mnt/data/submission.csv", index=False)
print("Submission file saved as submission.csv")




