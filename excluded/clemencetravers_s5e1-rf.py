import numpy as np # linear algebra
import pandas as pd # data processing, 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


# Load data
train_data=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_data=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


test_data


train_data.head()


train_data.info()


train_data.num_sold.describe()


sns.histplot(data=train_data['num_sold'])


train_data.dropna(subset=['num_sold'], inplace=True)


train_data.num_sold.isnull().sum()



 train_data['num_sold'] = train_data['num_sold'].apply(lambda x: 3000 if x > 3000 else x)


print(train_data['id'].corr(train_data.num_sold))


to_drop =['date']
#id


train_data.drop(to_drop,axis=1, inplace=True)


y=train_data.pop('num_sold')


X_train, X_val,y_train , y_val = train_test_split(train_data, y, random_state = 0)


object_cols=['country','store','product']


encoder =OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(encoder.fit_transform(X_train[object_cols]))
OH_cols_val = pd.DataFrame(encoder.transform(X_val[object_cols]))


# One-hot encoding removed index; put it back
OH_cols_train.index = OH_cols_train.index
OH_cols_val.index = OH_cols_val.index
# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_val = X_val.drop(object_cols, axis=1)
# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_val = pd.concat([num_X_val, OH_cols_val], axis=1)
# Ensure all columns have string type
#OH_X_train.columns = OH_X_train.columns.astype(str)


metric= mean_absolute_percentage_error


model_1 = RandomForestClassifier(n_estimators=100)

model_1.fit(OH_cols_train, y_train)

val_predictions = model_1.predict(OH_cols_val)
print(metric(y_val, val_predictions))


test_data.drop(to_drop,axis=1, inplace=True)



test_data.set_index('id', inplace=True)


OH_cols_test = pd.DataFrame(encoder.fit_transform(test_data[object_cols]))


# One-hot encoding removed index; put it back
OH_cols_test.index = test_data.index

# Remove categorical columns (will replace with one-hot encoding)
#num_X_test = test_data.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
#OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
OH_cols_test.columns = OH_cols_test.columns.astype(str)


OH_cols_test


prediction =model_1.predict(OH_cols_test)


output = pd.DataFrame({ 'id':test_data.index,
                       'num_sold': prediction})


output.set_index('id')


output.to_csv('submission_1.csv', index=False)

