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


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,confusion_matrix
from xgboost import XGBClassifier



training_data=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
testing_data=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


training_data.head()


training_data.shape


training_data.isnull().mean()*100


training_data=training_data.drop(["id"],axis=1)


training_data.info()


col=training_data.columns


for i in col:
    print(training_data[i].value_counts())


sns.heatmap(training_data[['Time_spent_Alone','Social_event_attendance',
       'Going_outside', 'Friends_circle_size','Post_frequency']].corr())


training_data.columns


le=LabelEncoder()


x=training_data.drop(["Personality"],axis=1)
y=le.fit_transform(training_data["Personality"])
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2)


num_col=['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
cat_col=['Stage_fear','Drained_after_socializing']


num_pipe=Pipeline([("num_impute",SimpleImputer(strategy="most_frequent"))])
cat_pipe=Pipeline([("cat_imput",SimpleImputer(strategy="most_frequent")),
                  ("ohe",OneHotEncoder(handle_unknown="ignore",drop="first"))])
        


col_tra=ColumnTransformer([("num_pipe",num_pipe,num_col),
                          ("cat_pipe",cat_pipe,cat_col)],remainder="passthrough")


model=XGBClassifier(n_estimators= 1100,
    max_depth= 3)


pipe = Pipeline([
    ('preprocess', col_tra),
    ('classifier', model)
])


pipe.fit(x_train,y_train)


pr=pipe.predict(x_val)
accuracy_score(y_val,pr)


testing_data=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


test=testing_data.drop(["id"],axis=1)


pred=pipe.predict(test)
pred = le.inverse_transform(pred)


submission = pd.DataFrame({'id': testing_data['id'], 'Personality': pred})
submission.to_csv('submission.csv', index=False)
print("Submission File saved!!")




