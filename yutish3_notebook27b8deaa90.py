# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv")
test = pd.read_csv("/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv")
specimen = pd.read_csv("../input/recruitment-task-for-gdsc-ml/SPECIMEN.csv")


train.info()


column_names = train.columns
print(column_names)


train.columns = train.columns.str.strip()
train.columns = train.columns.str.replace(" ", "_")
train.columns = train.columns.str.replace("'", "")
train.columns = train.columns.str.replace("\\", "")
train.columns = train.columns.str.replace(",", "")
train.columns = train.columns.str.replace("<", "")
train.columns = train.columns.str.replace(">", "")
train.columns = train.columns.str.replace("<", "")
train.columns = train.columns.str.replace(">", "")
test.columns = test.columns.str.strip()
test.columns = test.columns.str.replace(" ", "_")
test.columns = test.columns.str.replace("'", "")
test.columns = test.columns.str.replace("\\", "")
test.columns = test.columns.str.replace(",", "")
test.columns = test.columns.str.replace("<", "")
test.columns = test.columns.str.replace(">", "")
train.columns = train.columns.str.replace("[", "")
train.columns = train.columns.str.replace("]", "")


train['maT_r'] = train['maT_r'].fillna('nan')
train["F3Ku"] = train["F3Ku"].fillna('nan')
train['MINDSPIKE_VERSION'] = train['MINDSPIKE_VERSION'].fillna('nan')


train_encoded = pd.get_dummies(train, columns=['maT_r', "F3Ku", 'MINDSPIKE_VERSION'], drop_first=True)
test_encoded = pd.get_dummies(test, columns=['maT_r', "F3Ku", 'MINDSPIKE_VERSION'], drop_first=True)

train_labels = train_encoded.columns
test_labels = test_encoded.columns

missing_in_test = set(train_labels) - set(test_labels)

for c in missing_in_test:
    test_encoded[c] = 0

missing_in_train = set(test_labels) - set(train_labels)

test_encoded = test_encoded.drop(columns=list(missing_in_train))

test_encoded = test_encoded[train_labels]


train_encoded.info()


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

numerical_cols = train_encoded.select_dtypes(include=np.number).columns.tolist()

numerical_cols.remove('LOCAL_IDENTIFIER')

if 'CORRUCYSTIC_DENSITY' in numerical_cols:
    numerical_cols.remove('CORRUCYSTIC_DENSITY')

imputer = IterativeImputer(max_iter=10, random_state=0)

train_encoded[numerical_cols] = imputer.fit_transform(train_encoded[numerical_cols])

test_encoded[numerical_cols] = imputer.transform(test_encoded[numerical_cols])



train_encoded.info()


train_full = train_encoded.dropna(subset=['CORRUCYSTIC_DENSITY']).reset_index(drop=False)


from sklearn.ensemble import RandomForestRegressor

features = [
    'Z~x0k','vzo."','+U@','A.','hp!','?64:','@wnskR','U"r','&%)LTaWRb','r1Ng','|G}','TSWm','r2Ng','@V9','T!','14W$Q','ZZw3=!t','.om','.b6nl','!!','~7*','9Z/5)2',
    '%IiL7w','!;@Jw','fPqsI','ZVf','i7V','Jvi',';"i(T','Kj','w-u:jNqI','PZ8','jNhEum','xq','v0rt3X','^%a;','b1oRb13','v1rt3X',
    '0HU2N=U','ZrK','.6AvGp','3Iy','b2oRb13','maT_r_corro','F3Ku_qou'
]

X = pd.get_dummies(train_full[features])

y = train_full["CORRUCYSTIC_DENSITY"]

model = RandomForestRegressor(n_estimators=400, max_depth=7, random_state=1)
model.fit(X, y) 

X_test = pd.get_dummies(test_encoded[features])

predictions = model.predict(X_test)



submission = specimen.copy()
submission["CORRUCYSTIC_DENSITY"] = predictions

submission.to_csv("submission.csv", index=False)
print(submission.head())


