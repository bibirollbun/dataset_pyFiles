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


import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 


Sample_Submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
train_data = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
from sklearn.model_selection import train_test_split 


raw_id = test_data["id"]


Sample_Submission


raw_id


test_data


df = train_data 


df


df.isna().sum()


df.shape



imputer = SimpleImputer()


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


df["num_sold"] = imputer.fit_transform(df[["num_sold"]])


df["num_sold"].isna().sum()


df.duplicated().sum()


df.info()


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['date'] = le.fit_transform(df['date'])
df['country'] = le.fit_transform(df['country'])
df['store'] = le.fit_transform(df['store'])
df['product'] = le.fit_transform(df['product'])

test_data['date'] = le.fit_transform(test_data['date'])
test_data['country'] = le.fit_transform(test_data['country'])
test_data['store'] = le.fit_transform(test_data['store'])
test_data['product'] = le.fit_transform(test_data['product'])


df


# df['date'] = pd.to_datetime(df['date'])


df.info()


plt.figure(figsize=(10,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, fmt='.2f')
plt.show()


correlation = df['id'].corr(df['date'])
print("Correlation:", correlation)


df


X = df.drop(columns = ["num_sold"])
y = df["num_sold"]


X



# X = X.drop(columns = ["date"])


scaler = StandardScaler()
X = scaler.fit_transform(X)

test_data = scaler.fit_transform(test_data)


X = pd.DataFrame(X)


X



x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)





from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


y_pred = pd.DataFrame(y_pred)


y_pred


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)



y_pred_new = model.predict(test_data)


y_pred_new = pd.DataFrame(y_pred_new)


y_pred_new


test_data = pd.DataFrame(test_data)
test_data


test_data_new = test_data.iloc[:,0]


test_data_new








combined_df = pd.concat([raw_id, y_pred_new], axis=1, ignore_index=False)


combined_df


df.rename(columns={ df.columns[1]: "num_sold" }, inplace = True)


combined_df


combined_df.to_csv('file1.csv')










