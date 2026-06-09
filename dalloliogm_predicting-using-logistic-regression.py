import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



!head /kaggle/input/playground-series-s5e3/sample_submission.csv
!head /kaggle/input/playground-series-s5e3/train.csv
!head /kaggle/input/playground-series-s5e3/test.csv


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")#.set_index("id")
train.head()


train.describe()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")#.set_index("id")
test.head()


train.isnull().sum()



test.isnull().sum()



test[test.isnull().any(axis=1)]



from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define feature columns
features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 
            'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

X = train[features]
y = train['rainfall']

pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X, y)

# Predict probabilities on the training set
train_preds = pipeline.predict_proba(X)[:, 1]
print("Training AUC:", roc_auc_score(y, train_preds))


X_test = test[features]
X_test


# Predict probabilities on the test set
test_preds = pipeline.predict_proba(X_test)[:, 1]

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': test_preds
})

# Save the submission file
submission.to_csv('submission.csv', index=False)


