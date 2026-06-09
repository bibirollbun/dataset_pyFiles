# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
import warnings


df = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')
t = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


df.columns = df.columns.str.strip()
t.columns = t.columns.str.strip()

# Ensure index alignment
df = df.reset_index(drop=True)
t = t.reset_index(drop=True)

# Concatenate along columns
train = pd.concat([df, t], axis=1)

# Handle duplicate columns: Take first non-null value for each duplicate column
train = train.groupby(train.columns, axis=1).first()

# Verify result
print(train.info())


test.info()


test["winddirection"] = test["winddirection"].fillna(test["winddirection"].mean())


train["rainfall"] = train["rainfall"].map({"yes": 1, "no": 0, 1: 1, 0: 0})


train = train.drop(columns = ['id'],axis =1)
test = test.drop(columns = ['id'],axis =1)


fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20, 20))
axes = axes.flatten()

# Plot each column
for i, (col, values) in enumerate(train.items()):
    sns.histplot(values, ax=axes[i], kde=True)  # Replaces deprecated sns.distplot
    axes[i].set_title(col)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in train.items():
    sns.boxplot(y=col, data=train, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# Correlation matrix
correlation_matrix = train.corr()

# Heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


X = train.drop(['rainfall'],axis=1)
y = train['rainfall']



from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

def classify(model, X, y):
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    model.fit(X_train, y_train)
    print('roc_auc_score: ', model.score(X_test, y_test))  

    score = cross_val_score(model, X, y, cv=5)
    print('CV Score :', np.mean(score))


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify (model, X, y)



import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score

# Objective function for Optuna
def objective(trial):
    try:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"])
        C = trial.suggest_loguniform("C", 1e-4, 1e2)
        solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga", "newton-cg"])
        max_iter = trial.suggest_int("max_iter", 100, 5000, step=100)
        tol = trial.suggest_loguniform("tol", 1e-4, 1e-2)

        # Ensure solver-penalty compatibility
        if penalty == "none":
            solver = "lbfgs"  # Only lbfgs supports no penalty
        elif penalty == "l1" and solver not in ["liblinear", "saga"]:
            solver = "liblinear"
        elif penalty == "elasticnet" and solver != "saga":
            solver = "saga"

        l1_ratio = trial.suggest_float("l1_ratio", 0, 1) if penalty == "elasticnet" else None

        # Define model
        model = LogisticRegression(
            penalty=None if penalty == "none" else penalty,  
            C=C, 
            solver=solver, 
            max_iter=max_iter, 
            tol=tol, 
            l1_ratio=l1_ratio if penalty == "elasticnet" else None
        )

        # Cross-validation with ROC AUC scoring
        score = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()
        return score

    except Exception as e:
        print(f"Trial failed due to error: {e}")
        return float("-inf")  # Return a very low score for failed trials

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print best results safely
if study.best_trial:
    print("Best hyperparameters:", study.best_trial.params)
    print("Best ROC AUC score:", study.best_trial.value)
else:
    print("No valid trials found.")



train_features = train.drop(columns=["rainfall"], errors="ignore").columns

# Ensure test has only those columns, in the correct order
test = test[train_features]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Best hyperparameters found by Optuna
best_params = {
    'penalty': 'l2',
    'C': 21.368193173027922,
    'solver': 'liblinear',
    'max_iter': 1800,
    'tol': 0.00021195639021872196
}

# Train the final model on the full training data
final_model = LogisticRegression(**best_params)
final_model.fit(X, y)  # Assuming X_train, y_train are defined

# Predict probabilities for ROC AUC evaluation
y_pred_prob = final_model.predict_proba(test)[:, 1]  # Get probabilities for the positive class





submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission['rainfall'] = y_pred_prob


submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")




