# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/final-training-dataset/df_train_final.csv')
df.head()


from bayes_opt import BayesianOptimization

# Drop participant_id
df.drop(columns=["participant_id"], inplace=True)

# Define features and targets
X = df.drop(columns=["ADHD_Outcome", "Sex_F"])
y_adhd = df["ADHD_Outcome"]
y_sex = df["Sex_F"]

# Split data into training and testing sets
X_train_adhd, X_test_adhd, y_train_adhd, y_test_adhd = train_test_split(X, y_adhd, test_size=0.2, random_state=42)
X_train_sex, X_test_sex, y_train_sex, y_test_sex = train_test_split(X, y_sex, test_size=0.2, random_state=42)

# Define Bayesian Optimization function
def lgb_evaluate(num_leaves, learning_rate, n_estimators, max_depth, X_train, y_train, X_test, y_test):
    params = {
        'num_leaves': int(num_leaves),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'objective': 'binary',
        'verbosity': -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

# Bayesian Optimization
bounds = {
    'num_leaves': (20, 60),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (50, 300),
    'max_depth': (3, 15)
}

def train_lightgbm(X_train, y_train, X_test, y_test):
    optimizer = BayesianOptimization(
        f=lambda num_leaves, learning_rate, n_estimators, max_depth: lgb_evaluate(
            num_leaves, learning_rate, n_estimators, max_depth, X_train, y_train, X_test, y_test
        ),
        pbounds=bounds,
        random_state=42,
        verbose=0
    )
    optimizer.maximize(init_points=5, n_iter=20)
    best_params = optimizer.max['params']
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    
    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(classification_report(y_test, y_pred))
    
    return best_model, accuracy, f1, y_pred

# Train and evaluate models
best_model_adhd, acc_adhd, f1_adhd, y_pred_adhd = train_lightgbm(X_train_adhd, y_train_adhd, X_test_adhd, y_test_adhd)
best_model_sex, acc_sex, f1_sex, y_pred_sex = train_lightgbm(X_train_sex, y_train_sex, X_test_sex, y_test_sex)

# Create a DataFrame with actual vs predicted values
output_df = pd.DataFrame({
    "Actual_ADHD": y_test_adhd.values,
    "Predicted_ADHD": y_pred_adhd,
    "Actual_Sex": y_test_sex.values,
    "Predicted_Sex": y_pred_sex
})

# Save to CSV
output_df.to_csv("model_predictions.csv", index=False)
print("Predictions saved to model_predictions.csv")

# Print results
print(f"ADHD Model - Accuracy: {acc_adhd:.4f}, F1 Score: {f1_adhd:.4f}")
print(f"Sex Model - Accuracy: {acc_sex:.4f}, F1 Score: {f1_sex:.4f}")

