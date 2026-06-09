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


from sklearn.compose import make_column_transformer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd

train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

# Define features (X) and target (y)
X = train.drop(columns=['id', 'target', 'bin_0'])
y = train['target']

# Define the preprocessor for OneHotEncoding categorical variables
preprocessor = make_column_transformer(
    (OneHotEncoder(categories='auto', sparse_output=True, dtype='uint8', handle_unknown="ignore"), [f for f in X.columns]),  # Exclude target' columns
    remainder='passthrough'  # Keep the other columns (numeric columns) as they are
)

# Create a pipeline with preprocessor and logistic regression
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logreg', LogisticRegression(C= 0.123456789, max_iter=500))
])

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

# Perform cross-validation and get the scores
scores = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc", return_train_score=True)
cv_score = scores["test_score"].mean()

print(f"Cross-validation AUC score: {cv_score:.7f}")


# Load test set
test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
X_test_final = test.drop(columns=['id', 'bin_0'])

# Fit Model
pipe.fit(X, y)

# Make predictions with probabilities (roc_auc is based on probabilities)
y_pred_proba_test = pipe.predict_proba(X_test_final)[:, 1]  # Get probability of class 1

# Prepare the submission dataframe with the ID and predicted target (probability)
submission = pd.DataFrame({'id': test['id'], 'target': y_pred_proba_test})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)

