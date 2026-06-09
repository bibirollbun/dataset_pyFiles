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


train= pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train.columns


test= pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test.columns


train.describe()


train.head()


train.info()


X_train= train.drop(columns=['id', 'y'])
y_train= train['y']
x_test= test.drop(columns=['id'])


cat_col= X_train.select_dtypes(include='object').columns
print(cat_col.tolist())


from sklearn.preprocessing import OneHotEncoder

ohe= OneHotEncoder(drop= 'first', sparse_output= False, handle_unknown= 'ignore')


X_train_ohe= ohe.fit_transform(X_train[cat_col])



x_test_ohe= ohe.transform(x_test[cat_col])


ohe_feature_names= ohe.get_feature_names_out(cat_col)

X_train_ohe= pd.DataFrame(X_train_ohe, columns= ohe_feature_names, index= X_train.index)
x_test_ohe = pd.DataFrame(x_test_ohe, columns=ohe_feature_names, index=x_test.index)


X_train= X_train.drop(columns= cat_col)
x_test= x_test.drop(columns= cat_col)
X_train_pro= pd.concat([X_train, X_train_ohe], axis=1)
x_test_pro= pd.concat([x_test, x_test_ohe], axis=1)


print(X_train.select_dtypes(include= 'object').columns)
print(x_test.select_dtypes(include= 'object').columns)


# After creating X_train_pro and X_test_pro:
missing_cols = set(X_train_pro.columns) - set(x_test_pro.columns)
for col in missing_cols:
    x_test_pro[col] = 0  # Add missing OHE columns with 0s

# Ensure identical column order
x_test_pro = x_test_pro[X_train_pro.columns]


import seaborn as sns
import matplotlib.pyplot as plt

y_train.value_counts()


corr= X_train_pro.corr(numeric_only= True)
plt.figure(figsize=(30,30))
sns.heatmap(corr, annot= True, cmap= 'coolwarm')


sns.histplot(X_train_pro['age'], kde= True)
plt.title('Age distribution')


from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier(class_weight= 'balanced', n_estimators= 200, max_depth= 10)

rfc.fit(X_train_pro, y_train)


y_pred= rfc.predict(x_test_pro)
y_proba = rfc.predict_proba(x_test_pro)[:, 1]


print(test.columns)


test_ids = test['id']

# Verify alignment
assert len(test_ids) == len(x_test_pro), "Row count mismatch!"


submission= pd.DataFrame({
    'id': test_ids,
    'y': y_proba
})


print(submission.head())


submission.to_csv("submission.csv", index= False)




