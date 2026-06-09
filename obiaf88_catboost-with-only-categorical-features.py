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


from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head(2)


test.head(2)


train.shape, test.shape


num_columns =[col for col in train.select_dtypes(include = np.number).columns if col not in ['id']]
cat_columns = [col for col in train.select_dtypes(include = 'object').columns if col not in ['Fertilizer Name'] ]


poly = PolynomialFeatures(degree = 2, interaction_only = True,include_bias=False)


train_num = pd.DataFrame(poly.fit_transform(train[num_columns]), columns = poly.get_feature_names_out())
test_num = pd.DataFrame(poly.transform(test[num_columns]), columns = poly.get_feature_names_out())


train_categorical = pd.DataFrame()
test_categorical = pd.DataFrame()


for col in train_num.columns:
    train_categorical[f'{col}_quantile'] = pd.qcut(train_num[col],10).astype('str')
    test_categorical[f'{col}_quantile'] = pd.qcut(test_num[col],10).astype('str')
    


test_categorical.head(2)


test_categorical.head(2)


train_categorical = pd.concat([train_categorical,train[['Soil Type', 'Crop Type','Fertilizer Name']]], axis = 1)
test_categorical = pd.concat([test_categorical,test[cat_columns]], axis = 1)


train_categorical.shape, test_categorical.shape


test_categorical.info()


X = train_categorical[[col for col in train_categorical.columns if col not in 'Fertilizer Name']]
y = pd.get_dummies(train['Fertilizer Name'])


clf = CatBoostClassifier(
    loss_function='MultiClass',
    cat_features = list(range(X.shape[1])),
    verbose = 0,
    learning_rate = 0.1,
    iterations  = 1000
)


clf.fit(X,train['Fertilizer Name'])


ordered_classes = clf.classes_[np.argsort(-clf.predict_proba(test_categorical))[:,:3]]


ordered_classes


res = []


for i in range(ordered_classes.shape[0]):
    res.append((test['id'].iloc[i], ' '.join(ordered_classes[i])))


submission = pd.DataFrame(res,columns = ['id','Fertilizer Name'])


submission


submission.to_csv('submission.csv', index=False)
print("Submission created")

