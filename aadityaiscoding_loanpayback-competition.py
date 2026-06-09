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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test =  pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


train.head()


train['loan_paid_back'].value_counts(normalize=True) #checking imbalance clases
 


num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
cat_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']


# encoding processing pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])



from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Split data
X = train.drop(['loan_paid_back', 'id'], axis=1)
y = train['loan_paid_back']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




# Create pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, solver='saga'))
])
# Train
clf.fit(X_train, y_train)




# Predict
y_pred = clf.predict(X_val)
y_pred_prob = clf.predict_proba(X_val)[:, 1]

# Evaluate
print(classification_report(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_pred_prob))



# Drop id temporarily
X_test = test.drop(['id'], axis=1)

# Predict probabilities
test_probs = clf.predict_proba(X_test)[:, 1]

# submission dataframe
submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_probs
})




# to CSV
submission.to_csv("submission.csv", index=False)

submission.head()


from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'model__C': loguniform(1e-3, 1e2),    # sample C values between 0.001 and 100
    'model__penalty': ['l1', 'l2'],       # L1 = Lasso-like, L2 = Ridge-like
}

# Randomized search setup
rand_search = RandomizedSearchCV(
    clf,
    param_distributions=param_distributions,
    n_iter=20,                # number of random combinations to try
    scoring='roc_auc',        # optimize ROC-AUC
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

# Fit search
rand_search.fit(X_train, y_train)


print("Best parameters:", rand_search.best_params_)
print("Best cross-val ROC-AUC:", rand_search.best_score_)

# Evaluate on validation set
best_model = rand_search.best_estimator_
y_val_pred = best_model.predict(X_val)
y_val_prob = best_model.predict_proba(X_val)[:, 1]

from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_val, y_val_pred))
print("Validation ROC-AUC:", roc_auc_score(y_val, y_val_prob))

