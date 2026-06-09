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


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head(10)


test.isnull().sum()


test.head()


y = train['rainfall']


X = train.drop(['id', 'rainfall'], axis=1)


from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer


model = XGBClassifier(
        objective="binary:logistic", 
        eval_metric="logloss", 
        use_label_encoder=False,
        tree_method="hist",
        device="gpu"
    )


model.fit(X, y)


imputer = SimpleImputer()


test['winddirection'] = imputer.fit_transform(test['winddirection'].values.reshape(-1, 1))


preds = model.predict(test.drop('id', axis=1))


sub = pd.DataFrame({
    'id': test['id'].values,
    'rainfall': preds
})


sub.to_csv('submission.csv', index=False)







