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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score ,StratifiedKFold , train_test_split , GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df.info()


df.head()


print(df['marital_status'].unique()) 
print('--------------')
print(df['education_level'].unique())
print('---------------')
print(df['employment_status'].unique())
print('--------------')
print(df['loan_purpose'].unique())
print('--------------')
print(df['grade_subgrade'].unique())



plt.figure(figsize=(8,5))
df["annual_income"].plot(kind="kde")
plt.title("Densité de probabilité de l'annual_income")
plt.xlabel("Annual Income")
plt.ylabel("Densité de probabilité")
plt.grid(True)
plt.show()


df["debt_to_income_ratio"].plot(kind="kde")
plt.title("Densité de probabilité du debt_to_income_ratio")
plt.xlabel("debt_to_income_ratio")
plt.ylabel("Densité de probabilité")
plt.grid(True)
plt.show()

df["credit_score"].plot(kind="kde")
plt.title("Densité de probabilité du credit_score")
plt.xlabel("credit_score")
plt.ylabel("Densité de probabilité")
plt.grid(True)
plt.show()


df["loan_amount"].plot(kind="kde")
plt.title("Densité de probabilité du loan_amount")
plt.xlabel("loan_amount")
plt.ylabel("Densité de probabilité")
plt.grid(True)
plt.show()



df["interest_rate"].plot(kind="kde")
plt.title("Densité de probabilité du interest_rate")
plt.xlabel("interest_rate	")
plt.ylabel("Densité de probabilité")
plt.grid(True)
plt.show()



columns = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]

encoder = LabelEncoder()
for col in columns:
    df[col] = encoder.fit_transform(df[col])

df.head()


X = df.drop(['loan_paid_back', 'id'], axis=1)
y = df['loan_paid_back']


X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2,      
    train_size=0.8,     
    random_state=42,    
    stratify=y          
)




xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    device="cuda"
)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


for metric in scoring:
    scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring=metric)
  
    metric_name = metric.replace('_', ' ').upper()
     
    print(f"{metric_name}:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Individual fold scores: {scores}")
    print()


xgb_model.fit(X_train, y_train)


val_score = xgb_model.score(X_val, y_val)


print(f"Validation accuracy: {val_score}")


xgb = XGBClassifier(random_state=42, eval_metric='logloss', device="cuda")

param_grid = {
    'n_estimators': [100, 200, 300],           # Nombre d'arbres
    'max_depth': [3, 5, 7, 9],                 # Profondeur des arbres
    'learning_rate': [0.01, 0.05, 0.1, 0.3],   # Taux d'apprentissage
    'gamma': [0, 0.1]                     # Régularisation
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='roc_auc',              # Métrique à optimiser (ou 'accuracy', 'f1', etc.)
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=2,                       # Affiche la progression
    n_jobs=-1                        # Utilise tous les CPU
)


grid_search.fit(X_train, y_train)
print(grid_search.best_params_)



print(f"MEILLEUR SCORE (CV): {grid_search.best_score_:.4f}")


