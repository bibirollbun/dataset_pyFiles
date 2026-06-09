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

df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


pd.set_option('display.max_columns', None)


df.head()
#one hot encoding : gender, ethnicity, employment_status, smoking_status 
#label encoding : education_level, income_level 


one_hot_columns = ['gender', 'ethnicity', 'smoking_status', 'employment_status']
df = pd.get_dummies(df, columns = one_hot_columns, drop_first = False)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)


df.head()


education_order = {
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}

df['education_level'] = df['education_level'].map(education_order)

income_order = {
    'Low' : 0,
    'Lower-Middle' : 1,
    'Middle' : 2,
    'Upper-Middle' : 3,
    'High' : 4
}

df['income_level'] =df['income_level'].map(income_order)


df.head()


pip install --upgrade scikit-learn imbalanced-learn



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

x = df.drop(['diagnosed_diabetes','id','waist_to_hip_ratio','cholesterol_total','bmi','ldl_cholesterol','hdl_cholesterol'], axis = 1)
y = df['diagnosed_diabetes']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

model = LGBMClassifier(n_estimators = 500,
                       learning_rate = 0.05,
                       max_depth = -1,
                       num_leaves = 20,
                       subsample = 0.8,
                       colsample_bytree = 0.8,
                       random_state = 42,
                       class_weight = 'balanced')

model.fit(x_train, y_train)

y_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

print(f"Accuracy : {accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, model.predict_proba(x_test)[:,1])
print("AUC:", auc)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


one_hot_columns = ['gender', 'ethnicity', 'smoking_status', 'employment_status']
df_test = pd.get_dummies(df_test, columns = one_hot_columns, drop_first = False)
bool_cols = df_test.select_dtypes(include="bool").columns
df_test[bool_cols] = df_test[bool_cols].astype(int)


education_order = {
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}

df_test['education_level'] = df_test['education_level'].map(education_order)

income_order = {
    'Low' : 0,
    'Lower-Middle' : 1,
    'Middle' : 2,
    'Upper-Middle' : 3,
    'High' : 4
}

df_test['income_level'] =df_test['income_level'].map(income_order)


df_test_final = df_test.drop(['id','waist_to_hip_ratio','cholesterol_total','bmi','ldl_cholesterol','hdl_cholesterol'],axis = 1)

y_prob = model.predict_proba(df_test_final)[:,1]
y_pred = (y_prob >= 0.5).astype(int)


submission = pd.DataFrame({
    'id' : df_test['id'],
    'diagnosed_diabetes' : y_pred
})

submission.to_csv('submission.csv', index = False)




