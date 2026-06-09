# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')


train.shape
test.shape


train.head()


test.head()


train=train.drop(['id'], axis=1)


train.describe()


train.info()


test.info()


test['winddirection'].fillna(test['winddirection'].median(), inplace=True)
print(test.isnull().sum())


corr_matrix = train.corr()
plt.figure(figsize=(12,7))
sb.heatmap(corr_matrix, annot=True, cmap='coolwarm',fmt=".2f")
plt.show()


# Separate features and target variable
X_train = train.drop(columns=['rainfall'])
y_train = train['rainfall']


# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=54)


# feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)


# Initialize models
log_reg = LogisticRegression()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()


# Train models
log_reg.fit(X_train_scaled, y_train)
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)


# Predictions on validation set
y_pred_log_reg = log_reg.predict(X_val_scaled)
y_pred_decision_tree = decision_tree.predict(X_val)
y_pred_random_forest = random_forest.predict(X_val)


# Classification reports
print("Logistic Regression Report:\n", classification_report(y_val, y_pred_log_reg))
print("Decision Tree Report:\n", classification_report(y_val, y_pred_decision_tree))
print("Random Forest Report:\n", classification_report(y_val, y_pred_random_forest))


# Comparing models using Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models = [("Logistic Regression", y_pred_log_reg), ("Decision Tree", y_pred_decision_tree), ("Random Forest", y_pred_random_forest)]
for i, (title, y_pred) in enumerate(models):
    sb.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
plt.show()


# Hyperparameter tunin
param_grid_log_reg = {"C": [0.01, 0.1, 1, 10]}
gs_log = GridSearchCV(LogisticRegression(), param_grid_log_reg, cv=5)
gs_log.fit(X_train_scaled, y_train)
best_log_model = gs_log.best_estimator_
print("Best Logistic Regression Params:", gs_log.best_params_)

param_grid_decision_tree = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
gs_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_decision_tree, cv=5)
gs_dt.fit(X_train, y_train)
best_dt_model = gs_dt.best_estimator_
print("Best Decision Tree Params:", gs_dt.best_params_)

param_grid_random_forest = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None]}
gs_rf = GridSearchCV(RandomForestClassifier(), param_grid_random_forest, cv=5)
gs_rf.fit(X_train, y_train)
best_rf_model = gs_rf.best_estimator_
print("Best Random Forest Params:", gs_rf.best_params_)


# Cross-validation scores
models = {"Logistic Regression": best_log_model, "Decision Tree": best_dt_model, "Random Forest": best_rf_model}
best_model_name = None
best_model = None
best_score = 0

print("Cross-validation Scores:")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"{name}: Mean={mean_score:.4f}, Std={scores.std():.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_model = model

print(f"Best Model: {best_model_name} with Score: {best_score:.4f}")


# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("Best model saved as best_model.pkl")


X_test = test.drop(['id'], axis=1)
X_test_scaled = scaler.transform(X_test)


# Make predictions on test set
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_dt = decision_tree.predict(X_test)
y_pred_best_rf = random_forest.predict(X_test)


#submission log_reg
submission = pd.DataFrame({"id": test['id'], "rainfall": y_pred_log})
submission.to_csv("submission_log.csv", index=False)
print("Predictions saved to submission_log.csv")
#submission dt
submission = pd.DataFrame({"id": test['id'], "rainfall": y_pred_dt})
submission.to_csv("submission_dt.csv", index=False)
print("Predictions saved to submission_dt.csv")
#submission rf
submission = pd.DataFrame({"id": test['id'], "rainfall": y_pred_best_rf})
submission.to_csv("submission_rf.csv", index=False)
print("Predictions saved to submission_rf.csv")




