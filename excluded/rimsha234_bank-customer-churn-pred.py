# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#  1. Load & Explore Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df_train = pd.read_csv("/kaggle/input/bank-customer-churn-prediction-challenge/train.csv", index_col="id")
df_test = pd.read_csv("/kaggle/input/bank-customer-churn-prediction-challenge/test.csv")

# Quick check
print(df_train.isnull().sum())
print(df_train['Exited'].value_counts(normalize=True))

# Define variables
cat_vars = ['Surname', 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
num_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']




## Feature Engineering

# WoE Encoding
def woe_category(column_data, unique=None):
    if unique is None:
        unique = column_data.value_counts() / len(column_data)
        unique = unique.apply(lambda x: np.log(x / (1 - x)))
    return column_data.map(unique), unique

cat_encoders = {}
for var in cat_vars[1:]:
    df_train[var + '_woe'], cat_encoders[var] = woe_category(df_train[var])

# Numeric transformations
numeric_transformations = {
    'Age': ['log_Age', np.log],
    'CreditScore': ['sqrt_CreditScore_2', np.power, 2]
}

for var, trans in numeric_transformations.items():
    if len(trans) == 3:
        df_train[trans[0]] = df_train[var].apply(lambda x: trans[1](x, trans[2]))
    else:
        df_train[trans[0]] = df_train[var].apply(trans[1])

# Final feature set
cat_model_vars = [var + '_woe' for var in cat_vars[1:]]
num_model_vars = [var for var in num_vars if var not in ['Age', 'CreditScore']] + ['log_Age', 'sqrt_CreditScore_2']




fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for i, col in enumerate(num_vars):  
    sns.boxplot(x='Exited', y=col, data=df_train, ax=axs[i], palette='Set2')  
    axs[i].set_title(f"{col} Boxplot by Churn")

plt.tight_layout()
plt.show()




## Train-Test Split
X = df_train[cat_model_vars + num_model_vars]
y = df_train['Exited']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

#  GridSearchCV LightGBM
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import time

# Class weights
class_weight_vals = compute_class_weight(class_weight='balanced', classes=[0,1], y=y)
class_weight = {0: class_weight_vals[0], 1: class_weight_vals[1]}

param_grid = {
    'n_estimators': [100, 500],
    'num_leaves': [31, 63],
    'max_depth': [-1, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

start_grid = time.time()
lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, class_weight=class_weight, verbosity=-1)
grid = GridSearchCV(lgbm, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
end_grid = time.time()

print("Best Parameters:", grid.best_params_)
print("ROC-AUC:", grid.best_score_)
print("Training Time (min):", (end_grid - start_grid)/60)




##  Evaluation
from sklearn.metrics import recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

y_pred = grid.predict(X_val)
rec = recall_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)
print("Recall:", rec)
print("Precision:", prec)
ConfusionMatrixDisplay(cm / cm.sum(), display_labels=grid.classes_).plot()
plt.show()




## Train model
grid.fit(X_train, y_train)
end_grid = time.time()

print("Best Parameters:", grid.best_params_)
print("ROC-AUC:", grid.best_score_)
print("Training Time (min):", (end_grid - start_grid)/60)


# Apply transformations to test set
for var, trans in numeric_transformations.items():
    if len(trans) == 3:
        df_test[trans[0]] = df_test[var].apply(lambda x: trans[1](x, trans[2]))
    else:
        df_test[trans[0]] = df_test[var].apply(trans[1])

for var in cat_vars[1:]:
    df_test[var + '_woe'] = df_test[var].map(cat_encoders[var])

# Prepare test features and make predictions
X_test = df_test[cat_model_vars + num_model_vars]
preds = grid.predict_proba(X_test)[:, 1]  

# Create submission file
submission = pd.DataFrame({
    'CustomerId': df_test['CustomerId'],  
    'Exited': preds
})

# Save to root directory
submission.to_csv('submission.csv', index=False)
print(" Submission file saved successfully.")



