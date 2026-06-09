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


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


df = df.drop(columns=['id','day'])


X = df.drop(columns=["rainfall"])
y = df["rainfall"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=23)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def objective(trial):
    # Select the classifier
    model_name = trial.suggest_categorical("classifier", [ "SVM", "GradientBoosting", "KNN", "AdaBoost"])
    
    # Define hyperparameter spaces
    if model_name == "SVM":
        C = trial.suggest_loguniform("C", 1e-3, 10)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    
    elif model_name == "GradientBoosting":
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.5)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

    elif model_name == "KNN":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    
    elif model_name == "AdaBoost":
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 1.0)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

    # Evaluate model using cross-validation
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
    return score

# Run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Get the best model and parameters
print("Best Model:", study.best_trial.params)


best_params = study.best_trial.params
if best_params["classifier"] == "SVM":
    model = SVC(C=best_params["C"], kernel=best_params["kernel"], probability=True, random_state=42)
elif best_params["classifier"] == "GradientBoosting":
    model = GradientBoostingClassifier(n_estimators=best_params["n_estimators"], 
                                       learning_rate=best_params["learning_rate"], random_state=42)
elif best_params["classifier"] == "KNN":
    model = KNeighborsClassifier(n_neighbors=best_params["n_neighbors"], weights=best_params["weights"])
elif best_params["classifier"] == "AdaBoost":
    model = AdaBoostClassifier(n_estimators=best_params["n_estimators"], 
                               learning_rate=best_params["learning_rate"], random_state=42)

model.fit(X_train, y_train)
print("Final Accuracy on Test Set:", model.score(X_test, y_test))



from sklearn.tree import DecisionTreeClassifier


def objective(trial):
    # Define hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
    max_depth = trial.suggest_int("max_depth", 1, 10)
    
    # Base classifier
    base_estimator = DecisionTreeClassifier(max_depth=max_depth)

    # Define AdaBoost model
    model = AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    # Perform cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

    return score  # Higher is better

study = optuna.create_study(direction="maximize")  # Maximize accuracy
study.optimize(objective, n_trials=50)  # Run for 50 trials

print("Best hyperparameters:", study.best_params)
print("Best cross-validation accuracy:", study.best_value)


best_params = study.best_params

# Define best base estimator
best_base_estimator = DecisionTreeClassifier(max_depth=best_params["max_depth"])

# Train final model
final_model = AdaBoostClassifier(
    base_estimator=best_base_estimator,
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    random_state=42
)

final_model.fit(X_train, y_train)

# Evaluate on test set
test_accuracy = final_model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")





