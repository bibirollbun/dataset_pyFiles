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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


train_df.head()


numeric_cols = ['Time_spent_Alone', 'Social_event_attendance','Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
import pandas as pd
from sklearn.impute import SimpleImputer

# For numeric columns
num_imputer = SimpleImputer(strategy='median')
train_df[numeric_cols] = num_imputer.fit_transform(train_df[numeric_cols])

# For categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])



train_df.isnull().sum()


from sklearn.preprocessing import OneHotEncoder
ode = OneHotEncoder(sparse_output=False)
encoded_cat = ode.fit_transform(train_df[categorical_cols])



df1=pd.DataFrame(encoded_cat,columns=ode.get_feature_names_out(categorical_cols))


train_df=train_df.drop(columns=categorical_cols,axis=1)


train_df=pd.concat([train_df,df1],axis=1)


train_df.head()


x = train_df.drop("Personality",axis=1)
y=train_df["Personality"]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train , val_X , train_y , val_y = train_test_split(x,y,random_state=1)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()


rf.fit(X_train,train_y)


y_pred = rf.predict((val_X))


from sklearn.metrics import mean_absolute_error


mean_absolute_error(y_pred,val_y)



rf.fit(x,y)


y_pred = rf.predict((val_X))


mean_absolute_error(y_pred,val_y)



test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


test_data.head()


numeric_cols = ['Time_spent_Alone', 'Social_event_attendance','Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
import pandas as pd
from sklearn.impute import SimpleImputer

# For numeric columns
num_imputer = SimpleImputer(strategy='median')
test_data[numeric_cols] = num_imputer.fit_transform(test_data[numeric_cols])

# For categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
test_data[categorical_cols] = cat_imputer.fit_transform(test_data[categorical_cols])



test_data.isnull().sum()


from sklearn.preprocessing import OneHotEncoder
ode = OneHotEncoder(sparse_output=False)
encoded_cat = ode.fit_transform(test_data[categorical_cols])



numeric_cols = ['Time_spent_Alone', 'Social_event_attendance','Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
import pandas as pd
from sklearn.impute import SimpleImputer

# For numeric columns
num_imputer = SimpleImputer(strategy='median')
test_data[numeric_cols] = num_imputer.fit_transform(test_data[numeric_cols])

# For categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
test_data[categorical_cols] = cat_imputer.fit_transform(test_data[categorical_cols])

from sklearn.preprocessing import OneHotEncoder
ode = OneHotEncoder(sparse_output=False)
encoded_cat = ode.fit_transform(test_data[categorical_cols])

df2=pd.DataFrame(encoded_cat,columns=ode.get_feature_names_out(categorical_cols))
test_df=test_data.drop(columns=categorical_cols,axis=1)
test_df=pd.concat([test_df,df2],axis=1)



features = test_df.columns


test_X = test_df[features]


test_pred = rf.predict(test_X)


le.classes_


label_map = {0: 'Extrovert', 1: 'Introvert'}
y_pred_labels = [label_map[i] for i in test_pred]



submission = pd.DataFrame({
    'id': test_data['id'],          # replace with your actual ID column
    'Personality': y_pred_labels
})

submission.to_csv('submission.csv', index=False)





