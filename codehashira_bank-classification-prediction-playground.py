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


from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
y = train.pop('y')
X = train
test.head()


cat_cols = X.select_dtypes(include=['object','category']).columns
for col in cat_cols:
    print(f'Cardinality for {col}: \n{train[col].value_counts()} \n\n')


pip install sweetviz


from IPython.display import IFrame
import sweetviz as sv

report = sv.analyze(train)
IFrame(src='./sweetviz_report.html', width=1000, height=600)


for col in cat_cols:
    le = LabelEncoder()
    X[col], test[col] = le.fit_transform(X[col]), le.fit_transform(test[col])


model = XGBClassifier()
kfold = StratifiedKFold(n_splits =5, shuffle = True, random_state = 42)
cross_val = cross_val_score(model, X,y, cv= kfold, error_score='raise', scoring='roc_auc')


test.head()


model.fit(X,y)
predictions = model.predict(test)

submission = pd.DataFrame({'id': test['id'],
                          'y': predictions})
submission.to_csv('submission.csv',index=False)

