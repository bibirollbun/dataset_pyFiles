import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Usefull Imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from matplotlib import pyplot as plt


# Loading the Dataset 
X_full = pd.read_csv('../input/playground-series-s5e11/train.csv') # full train data
X_test = pd.read_csv('../input/playground-series-s5e11/test.csv')

X_full.head() # a first glance


# Basic / primary Information
X_full.describe()


# Can be very usefull certain times
X_full.nunique()


# Creating a heat map using correlation matrix
train_num = X_full.select_dtypes(exclude = object)
label = list(train_num.columns)

plt.figure(figsize=(16, 6))
plt.imshow(train_num.corr(), cmap = 'tab20c')
plt.xticks(ticks=range(len(label)), labels=label, rotation=90)
plt.yticks(ticks=range(len(label)), labels=label)

plt.colorbar()

plt.show()


# extracting id and target
y = X_full.pop('loan_paid_back') # target
testID = X_test.pop('id') # usefull for end submission

X_full.drop('id', axis = 1, inplace = True)


# Categorical Cols
cat_cols = ['credit_score', 'gender', 'marital_status', 'debt_to_income_ratio', 
            'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

ord_cols = ['debt_to_income_ratio', 'credit_score', 'interest_rate'] # will be ordinally encoded


# Numerical Cols
num_cols = list(X_full.select_dtypes(include = np.number).columns)
num_cols


# Skiping this part currently
"""# Ordinal Encoding
Ord = OrdinalEncoder()
label_X_full = pd.DataFrame(Ord.fit_transform(X_full[ord_cols]))
label_X_test = pd.DataFrame(Ord.fit_transform(X_test[ord_cols]))

label_X_full.columns = ['one', 'two', 'three']
label_X_test.columns = ['one', 'two', 'three']

label_X_full.head(2)"""


# OneHotEncoding
Oh = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
X = pd.DataFrame(Oh.fit_transform(X_full[cat_cols]))
test = pd.DataFrame(Oh.transform(X_test[cat_cols]))

X.head(2)


# final test train
X = X.join([X_full[num_cols]])
test = test.join([X_test[num_cols]])

X.columns = X.columns.astype(str)
test.columns = test.columns.astype(str)
X.head(2)


# Scaling the data
scale = StandardScaler()
X = pd.DataFrame(scale.fit_transform(X))
test = pd.DataFrame(scale.transform(test))


# Splitting data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Creating Model
model = LogisticRegression(C=1e-3, max_iter = 1000)


model.fit(X_train, y_train)
pred = model.predict_proba(X_val)[:,1]
roc_auc_score(y_val, pred)


cross_val_score(model, X, y, cv = 5, scoring = 'roc_auc').mean()


model.fit(X, y)


final = model.predict_proba(test)[:,1]


final = pd.DataFrame({'id':testID, 'loan_paid_back':final})
final.head()


final.to_csv('submission.csv', index = False)

