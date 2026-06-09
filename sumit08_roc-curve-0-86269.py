import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.head()


train.isnull().sum()


test.head()


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mode()[0])


plt.figure(figsize=(12,6))
sns.scatterplot(x="id",y="winddirection", data= test)


test.isnull().sum()


!pip install catboost


!pip install optuna


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import optuna


X = train.drop(columns=["id", "rainfall"])
y = train["rainfall"]
X_test_kaggle = test.drop(columns=["id"])


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test_kaggle = scaler.transform(X_test_kaggle)


poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X = poly.fit_transform(X)
X_test_kaggle = poly.transform(X_test_kaggle)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def objective(trial):
    param = {
        'C': trial.suggest_float('C', 0.01, 10.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l2']),
        'solver': 'lbfgs',
        'max_iter': 2000,  # Increased for better convergence
        'random_state': 42
    }
    model = LogisticRegression(**param)
    model.fit(X, y)
    y_probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_probs)
    return auc(fpr, tpr)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Increased trials for better hyperparameter search
print('Best trial:', study.best_trial.params)


model = LogisticRegression(**study.best_trial.params)
model.fit(X, y)


y_probs = model.predict_proba(X)[:, 1]



fpr, tpr, _ = roc_curve(y, y_probs)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
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

# ... your existing code ...

# Create the directory if it doesn't exist
os.makedirs("/mnt/data", exist_ok=True)

# Now save the dataframe
submission_df["rainfall"] = y_test_preds
submission_df.to_csv("/mnt/data/submission.csv", index=False)
print("Submission file saved as submission.csv")




