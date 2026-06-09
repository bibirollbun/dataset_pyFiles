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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from sklearn.metrics import accuracy_score,root_mean_squared_error



train_data=pd.read_csv("/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/train.csv")
test_data=pd.read_csv("/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/test.csv")
sample_data=pd.read_csv("/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/sample_submission.csv")


train_data.tail()


train_data['Exited'].value_counts()


test_data.head()


sample_data.isnull().sum()


train_data.isnull().sum()


test_data.isnull().sum()


train_data.info()


float_data=train_data.select_dtypes(include='float')
plt.figure(figsize=(10,10))
sns.heatmap(float_data.corr(),annot=True,cmap='coolwarm', fmt='.2f', cbar=True)
plt.show()


for col in train_data.select_dtypes(include='float').columns:
  sns.boxplot(train_data[col])
  plt.show()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train_data['Geography']=le.fit_transform(train_data['Geography'])
train_data['Gender']=le.fit_transform(train_data['Gender'])
train_data['Surname']=le.fit_transform(train_data['Surname'])


x_train = train_data.drop(columns=['Exited'],axis=1)
y_train = train_data['Exited']


x_train.shape,y_train.shape


from sklearn.model_selection import train_test_split
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.2, random_state=42)




from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_split=sc.fit_transform(x_train_split)
x_val_split=sc.transform(x_val_split)


from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(x_train_split,y_train_split)
y_pred=model.predict(x_val_split)
rmse = np.sqrt(np.mean((y_val_split - y_pred)**2))

print("RMSE from Gradient Boosting Classifier: {}".format(rmse))
print("accuracy:",model.score(x_val_split,y_val_split))


